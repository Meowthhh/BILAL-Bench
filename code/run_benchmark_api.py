#!/usr/bin/env python3
"""Run BILAL-Bench with GPT-OSS-120B through the Groq Chat Completions API.

This is the API runner used for the hosted-model arm of BILAL-Bench.  It sends
exactly one user message containing the benchmark's neutral Bengali instruction
(no system message, exemplars, retrieval tools, or browsing configuration), uses
``openai/gpt-oss-120b`` on Groq, temperature 0, and bounded 512->1024 token
auto-extension.  The first six output columns remain annotation-workbook
compatible; additional columns record run provenance.

The Groq Playground URL is a web UI.  Programmatic requests are sent to:
    https://api.groq.com/openai/v1/chat/completions

Example:
    export GROQ_API_KEY=YOUR_KEY
    python code/run_benchmark_api.py \
        --input data/BILAL_Bench_benchmark.csv \
        --output responses_GPT-OSS-120B.csv
"""

import argparse
import csv
import os
import sys
import time

import requests

from run_benchmark import (
    TEMPLATES,
    DEFAULT_STOP_STRINGS,
    OUT_COLS,
    REQUIRED_INPUT_COLS,
    already_done,
    load_questions,
    partition_existing,
    rewrite_keeping_good,
    trim_at_stop,
)

BILAL_BENCH_PHASE2_V2 = True

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MODEL_LABEL = "GPT-OSS-120B"


def validate_resume_compatibility(path, args, model_label, endpoint):
    """Prevent mixing outputs generated with different API configurations."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if not rows:
        return

    required = {
        "model", "model_id", "backend", "temperature", "reasoning_effort",
        "endpoint", "condition", "max_new_tokens", "ceiling", "auto_extend",
        "trim_stop", "rep_penalty", "no_repeat_ngram",
    }
    missing = sorted(required - fields)
    if missing:
        raise RuntimeError(
            f"Existing output is from an incompatible runner. Missing: {missing}. "
            f"Rename or remove {path}."
        )

    def first(key):
        for row in rows:
            value = str(row.get(key, "")).strip()
            if value and value.lower() != "nan":
                return value
        return ""

    def num_eq(a, b):
        try:
            return float(a) == float(b)
        except Exception:
            return str(a).strip() == str(b).strip()

    exact = {
        "model": model_label,
        "model_id": args.model,
        "backend": "groq",
        "reasoning_effort": args.reasoning_effort,
        "endpoint": endpoint,
    }
    for key, expected in exact.items():
        got = first(key)
        if got and got != str(expected):
            raise RuntimeError(
                f"Resume mismatch for {key}: existing={got!r}, requested={expected!r}. "
                f"Rename or remove {path}; do not mix configurations."
            )

    numeric = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "ceiling": args.ceiling,
        "auto_extend": 0 if args.no_auto_extend else 1,
        "trim_stop": 1 if args.trim_stop else 0,
        "rep_penalty": 1.0,
        "no_repeat_ngram": 0,
    }
    for key, expected in numeric.items():
        got = first(key)
        if got and not num_eq(got, expected):
            raise RuntimeError(
                f"Resume mismatch for {key}: existing={got!r}, requested={expected!r}. "
                f"Rename or remove {path}; do not mix configurations."
            )

    wanted = ({"neutral", "honesty-primed"} if args.condition == "both"
              else {args.condition})
    existing = {str(r.get("condition", "")).strip() for r in rows
                if r.get("condition")}
    if not existing.issubset(wanted):
        raise RuntimeError(
            f"Resume mismatch for condition: existing={sorted(existing)}, "
            f"requested={sorted(wanted)}."
        )


def call_groq(model, prompt, max_tokens, temperature, key, retries=6,
              endpoint=GROQ_ENDPOINT, reasoning_effort="medium"):
    """Call Groq's OpenAI-compatible chat-completions endpoint."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
    }

    wait = 3.0
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint,
                json=body,
                timeout=240,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
            )
        except requests.RequestException as exc:
            if attempt == retries - 1:
                return None, f"{type(exc).__name__}: {exc}"
            time.sleep(wait)
            wait = min(wait * 2, 60)
            continue

        if response.status_code == 200:
            try:
                return response.json(), ""
            except ValueError as exc:
                return None, f"Invalid JSON response: {exc}"

        if response.status_code in (408, 429, 500, 502, 503, 529):
            retry_after = response.headers.get("retry-after")
            try:
                sleep_for = float(retry_after) if retry_after else wait
            except ValueError:
                sleep_for = wait
            print(
                f"      HTTP {response.status_code}; sleeping {sleep_for:.0f}s "
                f"(attempt {attempt + 1}/{retries})",
                flush=True,
            )
            time.sleep(sleep_for)
            wait = min(wait * 2, 60)
            continue

        return None, f"HTTP {response.status_code}: {response.text[:500]}"

    return None, "exhausted retries"


def generate_complete_api(model, question, template, start_tokens, key,
                          temperature=0.0, auto_extend=True, ceiling=1024,
                          trim_stop=True, endpoint=GROQ_ENDPOINT,
                          reasoning_effort="medium"):
    """Generate a complete response with the benchmark's bounded retry rule.

    A length-capped first attempt is regenerated deterministically with a larger
    completion budget (512 -> 1024 by default).  This mirrors the historical
    benchmark runner and records the extension count in the output CSV.
    """
    prompt = template.format(q=question)
    cap, extensions, total_seconds = start_tokens, 0, 0.0

    while True:
        t0 = time.time()
        data, err = call_groq(
            model, prompt, cap, temperature, key,
            endpoint=endpoint, reasoning_effort=reasoning_effort,
        )
        total_seconds += time.time() - t0
        if err:
            return "", round(total_seconds, 1), err, 0, cap, extensions, "error"

        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = (message.get("content") or "").strip()
            finish = choice.get("finish_reason", "")
            usage = data.get("usage") or {}
            n_new = int(usage.get("completion_tokens") or 0)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            return (
                "", round(total_seconds, 1),
                f"Malformed Groq response: {type(exc).__name__}: {exc}",
                0, cap, extensions, "error",
            )

        if not text:
            if finish == "length" and auto_extend and cap < ceiling:
                cap = min(cap * 2, ceiling)
                extensions += 1
                print(
                    f"      (blank content, {n_new} completion tokens -> retrying with {cap})",
                    flush=True,
                )
                continue
            return (
                "", round(total_seconds, 1),
                f"EMPTY: blank content (finish_reason={finish}, completion_tokens={n_new})",
                n_new, cap, extensions, "empty",
            )

        if trim_stop:
            trimmed, found = trim_at_stop(text, DEFAULT_STOP_STRINGS)
            if found:
                return (
                    trimmed, round(total_seconds, 1), "",
                    n_new, cap, extensions, "trim-stop",
                )

        hit_cap = finish == "length"

        # Match the local runner's historical first-attempt collapse check.
        if hit_cap and extensions == 0:
            bn = sum(1 for c in text if "\u0980" <= c <= "\u09FF")
            words = text.split()
            punct = sum(text.count(c) for c in "!*[]()#|~`")
            why = None
            if bn < max(20, len(text) * 0.20):
                why = f"only {bn} Bengali chars"
            elif len(set(words)) < 25:
                why = f"lexical collapse ({len(set(words))} distinct words)"
            elif punct > len(text) * 0.12:
                why = f"punctuation spam ({punct} marks)"
            if why:
                print(
                    f"      (cut at {n_new} tok, {why} -- degenerate, not extending)",
                    flush=True,
                )
                return (
                    text, round(total_seconds, 1), "",
                    n_new, cap, extensions, "degenerate-cap",
                )

        if not hit_cap:
            return text, round(total_seconds, 1), "", n_new, cap, extensions, "eos"
        if not auto_extend or cap >= ceiling:
            return text, round(total_seconds, 1), "", n_new, cap, extensions, "cap"

        cap = min(cap * 2, ceiling)
        extensions += 1
        print(f"      (cut at {n_new} tok -> retrying with {cap})", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", default="data/BILAL_Bench_benchmark.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--name", default=DEFAULT_MODEL_LABEL,
                    help="label written to the model column")
    ap.add_argument("--output", default=None)
    ap.add_argument("--api-key-env", default=GROQ_API_KEY_ENV,
                    help="environment variable containing the Groq API key")
    ap.add_argument("--base-url", default=None,
                    help="override Groq chat-completions endpoint (mainly for testing)")
    ap.add_argument(
        "--condition", choices=["neutral", "honesty-primed", "both"],
        default="neutral",
    )
    ap.add_argument("--categories", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--ceiling", type=int, default=1024)
    ap.add_argument("--no-auto-extend", action="store_true")
    ap.add_argument(
        "--trim-stop", dest="trim_stop", action="store_true", default=True,
        help="trim self-generated follow-up turns (default: on)",
    )
    ap.add_argument(
        "--no-trim-stop", dest="trim_stop", action="store_false",
        help="disable post-hoc self-generated-turn trimming",
    )
    ap.add_argument(
        "--no-retry-failed", action="store_true",
        help="treat blank/errored rows as final instead of regenerating them on the next run",
    )
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"],
        default="medium",
        help="GPT-OSS reasoning effort; Groq's documented GPT-OSS default is medium",
    )
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause between requests to respect rate limits")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"ERROR: input not found: {args.input}")
    if args.max_new_tokens <= 0 or args.ceiling <= 0:
        sys.exit("ERROR: token limits must be positive")
    if args.ceiling < args.max_new_tokens:
        sys.exit("ERROR: --ceiling must be >= --max-new-tokens")
    if args.temperature != 0.0:
        print(
            "WARNING: the paper protocol used temperature=0.0; changing it will not "
            "reproduce the reported generation setting.",
            flush=True,
        )

    endpoint = args.base_url or GROQ_ENDPOINT
    key = os.environ.get(args.api_key_env)
    if not key and not args.dry_run:
        sys.exit(f"ERROR: {args.api_key_env} not set")

    label = args.name or args.model
    safe_label = label.replace("/", "-").replace(" ", "-")
    out_path = args.output or f"responses_{safe_label}.csv"
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    try:
        all_rows, questions = load_questions(args.input, cats, args.limit)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    conditions = (["neutral", "honesty-primed"] if args.condition == "both"
                  else [args.condition])

    print(f"Input: {args.input}  ({len(all_rows)} benchmark rows; all eligible)")
    print(f"Backend: Groq  endpoint={endpoint}  key_env={args.api_key_env}")
    print(f"Model: {args.model}  -> label {label!r}")
    print(
        f"Selected: {len(questions)} questions x {len(conditions)} condition(s) "
        f"= {len(questions) * len(conditions)} generations"
    )
    print(
        f"Decoding: temperature={args.temperature}  reasoning_effort={args.reasoning_effort} "
        f"max_new_tokens={args.max_new_tokens} ceiling={args.ceiling} "
        f"auto_extend={not args.no_auto_extend} trim_stop={args.trim_stop}"
    )
    print("No system message, exemplars, retrieval tools, or browsing tools are supplied.")

    if not questions:
        print("Nothing selected; exiting.")
        return 0

    if args.dry_run:
        for row in questions[:5]:
            print(
                f"  {row['question_id']}  cat{row['category']}  "
                f"{row['question_bn'][:55]}..."
            )
        print(f"[DRY RUN] would write {out_path}")
        return 0

    try:
        validate_resume_compatibility(out_path, args, label, endpoint)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2

    prior_attempts = {}
    if args.no_retry_failed:
        done = already_done(out_path)
        if done:
            print(
                f"Resuming: {len(done)} recorded pairs will be skipped "
                "(--no-retry-failed)."
            )
    else:
        good, failed, fields = partition_existing(out_path)
        for row in failed:
            key_pair = (row.get("question_id", ""), row.get("condition", ""))
            prior_attempts[key_pair] = max(
                prior_attempts.get(key_pair, 0), int(row.get("attempt") or 1)
            )
        if failed:
            archive = out_path + ".failed.csv"
            new_archive = not os.path.exists(archive)
            with open(archive, "a", newline="", encoding="utf-8") as af:
                writer = csv.DictWriter(af, fieldnames=fields or OUT_COLS)
                if new_archive:
                    writer.writeheader()
                for row in failed:
                    writer.writerow({k: row.get(k, "") for k in (fields or OUT_COLS)})
            rewrite_keeping_good(out_path, good, fields or OUT_COLS)
            print(
                f"Retry pass: {len(good)} successful rows kept; {len(failed)} "
                f"blank/errored rows archived to {archive} and will be regenerated."
            )
        elif good:
            print(
                f"Resuming: {len(good)} successful pairs skipped; no failed rows to retry."
            )
        done = {(r.get("question_id", ""), r.get("condition", "")) for r in good}

    new_file = not os.path.exists(out_path) or os.path.getsize(out_path) == 0
    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=OUT_COLS)
    if new_file:
        writer.writeheader()
        f.flush()

    total = len(questions) * len(conditions)
    index = generated = capped = 0
    started = time.time()

    try:
        for row in questions:
            for condition in conditions:
                index += 1
                pair = (row["question_id"], condition)
                if pair in done:
                    continue

                text, seconds, error, n_new, final_cap, extensions, reason = (
                    generate_complete_api(
                        args.model,
                        row["question_bn"],
                        TEMPLATES[condition],
                        args.max_new_tokens,
                        key,
                        temperature=args.temperature,
                        auto_extend=not args.no_auto_extend,
                        ceiling=args.ceiling,
                        trim_stop=args.trim_stop,
                        endpoint=endpoint,
                        reasoning_effort=args.reasoning_effort,
                    )
                )

                truncated = int(reason in ("cap", "degenerate-cap"))
                capped += truncated
                generated += 1

                writer.writerow({
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "model": label,
                    "condition": condition,
                    "annotator_id": "",
                    "response_text": text,
                    "model_id": args.model,
                    "model_revision": "provider-managed",
                    "backend": "groq",
                    "temperature": args.temperature,
                    "reasoning_effort": args.reasoning_effort,
                    "quant": "Groq TruePoint/provider-managed",
                    "device_map": "",
                    "attn": "",
                    "endpoint": endpoint,
                    "question_bn": row.get("question_bn", ""),
                    "expected_type": row.get("expected_type", ""),
                    "gold_reference": row.get("gold_reference", ""),
                    "gold_note": row.get("gold_note", ""),
                    "gen_seconds": seconds,
                    "error": error,
                    "n_new_tokens": n_new,
                    "final_cap": final_cap,
                    "extensions": extensions,
                    "stopped": reason,
                    "truncated": truncated,
                    "max_new_tokens": args.max_new_tokens,
                    "ceiling": args.ceiling,
                    "auto_extend": 0 if args.no_auto_extend else 1,
                    "trim_stop": 1 if args.trim_stop else 0,
                    "rep_penalty": 1.0,
                    "no_repeat_ngram": 0,
                    "prompt_style": "chat(api)",
                    "attempt": prior_attempts.get(pair, 0) + 1,
                })
                f.flush()
                done.add(pair)

                tag = "ERR" if error else reason
                print(
                    f"[{index}/{total}] {row['question_id']} {condition:14s} "
                    f"{tag:15s} {seconds:6.1f}s {n_new:5d}tok"
                    f"{' +' + str(extensions) if extensions else ''}",
                    flush=True,
                )
                time.sleep(args.sleep)
    finally:
        f.close()

    print(f"\nDone in {(time.time() - started) / 60:.1f} min. Output: {out_path}")
    if generated:
        print(
            f"CAP/degenerate-cap rows among newly generated: "
            f"{capped}/{generated} ({100.0 * capped / generated:.1f}%)"
        )
    print(
        "First six columns remain annotation-workbook compatible. "
        "Provenance columns follow them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
