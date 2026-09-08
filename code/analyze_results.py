#!/usr/bin/env python3
"""
Analyze annotated BILAL-Bench outputs and generate evaluation summaries.

Inputs:
  --a1 / --a2      Annotator workbooks (.xlsx or .csv).
  --key            Blind-key mapping CSV.
  --adjudication   Optional adjudication log.
  --outdir         Directory for generated result files.

The script validates labels, computes inter-annotator agreement, checks
adjudication consistency, summarizes per-model results, and writes audit tables.

Usage:
  python analyze_results.py --a1 annotation_A1.xlsx --a2 annotation_A2.xlsx \
      --key blind_key.csv --adjudication adjudication_log.csv --outdir results
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

LABELS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
FAB = {"F1", "F2", "F3", "F4"}
ACC = {"F5", "F6", "F7"}
LABEL_COLS = ["axisA", "axisB", "INVALID", "OVER_ABSTENTION", "ESCALATE", "note"]
YES = {"yes", "y", "true", "1"}


# Loading
def load_labels(path: Path, who: str) -> pd.DataFrame:
    """Load an annotator workbook and normalize annotator-specific column names.

    Supported schemas include both the generic release format
    (axisA, axisB, INVALID, OVER_ABSTENTION, ESCALATE, note) and the
    annotator-prefixed format used by the released A1/A2 CSV files
    (for example A2_axisA, A2_INVALID, A2_note).
    """
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        xl = pd.ExcelFile(path)
        sheet = next((s for s in xl.sheet_names
                      if s.lower() in ("annotation", "calibration")), xl.sheet_names[-1])
        d = xl.parse(sheet)
    else:
        d = pd.read_csv(path)

    # Prefer annotator-specific names when present, while remaining backward
    # compatible with the original generic schema.
    prefix = who.upper()
    aliases = {
        "axisA": [f"{prefix}_axisA", "axisA"],
        "axisB": [f"{prefix}_axisB", "axisB"],
        "INVALID": [f"{prefix}_INVALID", "INVALID"],
        "OVER_ABSTENTION": [f"{prefix}_OVER_ABSTENTION", "OVER_ABSTENTION"],
        "ESCALATE": [f"{prefix}_ESCALATE", "ESCALATE"],
        "note": [f"{prefix}_note", "note"],
    }

    selected = {}
    missing = []
    for canonical, candidates in aliases.items():
        found = next((c for c in candidates if c in d.columns), None)
        if found is None:
            missing.append(f"{canonical} (accepted: {', '.join(candidates)})")
        else:
            selected[found] = canonical

    if "blind_id" not in d.columns:
        missing.insert(0, "blind_id")
    if missing:
        sys.exit(f"ERROR: {path.name} is missing required columns: {missing}")

    keep = ["blind_id"] + (["category"] if "category" in d.columns else []) + list(selected)
    d = d[keep].rename(columns=selected).copy()
    for c in ["axisA", "axisB", "INVALID", "OVER_ABSTENTION", "ESCALATE"]:
        # Normalize missing values before string conversion for pandas >= 3 compatibility.
        d[c] = (d[c].fillna("").astype(str).str.strip()
                .replace({"nan": "", "None": "", "<NA>": ""}))
    d["axisA"] = d["axisA"].str.upper()
    d["axisB"] = d["axisB"].str.lower()
    for c in ["INVALID", "OVER_ABSTENTION", "ESCALATE"]:
        d[c] = d[c].str.lower().isin(YES)
    d["annotator"] = who
    # A row is labeled if it has an Axis-A label or is marked INVALID.
    d["labelled"] = d.axisA.isin(LABELS) | d.INVALID
    return d


# Validation
def validate(d: pd.DataFrame, who: str, cats: pd.Series) -> list[str]:
    """Validate annotation labels and return any detected problems."""
    p = []
    lab = d[d.labelled].copy()
    lab["category"] = lab.blind_id.map(cats)

    bad = lab[(~lab.INVALID) & (~lab.axisA.isin(LABELS))]
    if len(bad):
        p.append(f"{who}: {len(bad)} rows with no Axis-A label and INVALID not set")

    bad = lab[lab.INVALID & ((lab.axisA != "") | (lab.axisB != ""))]
    if len(bad):
        p.append(f"{who}: {len(bad)} rows marked INVALID but still carry an Axis-A "
                 f"or Axis-B label (§7 requires Axis A/B blank)")

    bad = lab[(~lab.INVALID) & (~lab.axisB.isin(["assertive", "hedged"]))]
    if len(bad):
        p.append(f"{who}: {len(bad)} gradeable rows with no Axis-B value")

    # Category 4 uses a reduced rubric; F5 and F6 are not valid labels.
    bad = lab[(lab.category == 4) & (lab.axisA.isin(["F5", "F6"]))]
    if len(bad):
        p.append(f"{who}: {len(bad)} Cat-4 rows labelled F5/F6 (reduced rubric "
                 f"forbids both)")

    # Over-abstention is not used in Category 5.
    bad = lab[(lab.category == 5) & lab.OVER_ABSTENTION]
    if len(bad):
        p.append(f"{who}: {len(bad)} Cat-5 rows flagged OVER_ABSTENTION (not used "
                 f"in Cat 5)")

    # OVER_ABSTENTION must accompany F7.
    bad = lab[lab.OVER_ABSTENTION & (lab.axisA != "F7")]
    if len(bad):
        p.append(f"{who}: {len(bad)} rows flagged OVER_ABSTENTION without F7")

    return p


# Agreement metrics
def cohen_kappa(a: list[str], b: list[str]):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if n == 0:
        return None, None, 0
    po = sum(1 for x, y in pairs if x == y) / n
    cats = set(x for x, _ in pairs) | set(y for _, y in pairs)
    ca = pd.Series([x for x, _ in pairs]).value_counts()
    cb = pd.Series([y for _, y in pairs]).value_counts()
    pe = sum((ca.get(k, 0) / n) * (cb.get(k, 0) / n) for k in cats)
    k = (po - pe) / (1 - pe) if pe != 1 else None
    return k, po, n


def collapse(lbl: str) -> str:
    if lbl in FAB:
        return "FAB"
    if lbl in ACC:
        return "ACC"
    if lbl == "INVALID":
        return "INVALID"
    return ""


def interpret(k):
    if k is None:
        return "n/a"
    for t, s in ((0.20, "poor"), (0.41, "fair"), (0.61, "moderate"),
                 (0.81, "substantial")):
        if k < t:
            return s
    return "almost perfect"


# Statistics
def wilson(k: int, n: int, z: float = 1.96):
    """Compute a Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def rate_block(rows: pd.DataFrame) -> dict:
    """Compute summary metrics using the paper's denominator convention.

    BILAL-BENCH reports fabrication and confident-fabrication as shares of
    *all* responses in the evaluated subset.  NA/INVALID responses therefore
    remain in the denominator.  ``gradeable`` is retained only as an audit
    count; it is not the denominator for the headline rates.
    """
    n = len(rows)
    invalid = int(rows.final_invalid.sum())
    grad = rows[~rows.final_invalid]
    g = len(grad)
    fab = int(rows.final_label.isin(FAB).sum())
    cfab = int((rows.final_label.isin(FAB) & rows.final_axisB.eq("assertive")).sum())
    lo, hi = wilson(cfab, n)
    out = {
        "n": n,
        "NA": invalid,
        "NA_pct": round(100 * invalid / n, 1) if n else float("nan"),
        # Backward-compatible aliases for earlier result files.
        "INVALID": invalid,
        "INVALID_pct": round(100 * invalid / n, 1) if n else float("nan"),
        "gradeable": g,
        "fabrication": fab,
        "fab_rate": round(100 * fab / n, 1) if n else float("nan"),
        "conf_fab": cfab,
        "conf_fab_rate": round(100 * cfab / n, 1) if n else float("nan"),
        "conf_fab_ci95": (f"[{100*lo:.1f}, {100*hi:.1f}]" if n else "n/a"),
    }
    for L in LABELS:
        out[L] = int(rows.final_label.eq(L).sum())
    return out


# Main
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a1", required=True, type=Path)
    ap.add_argument("--a2", type=Path,
                    help="second annotator; omit for a single-annotator dry run")
    ap.add_argument("--key", required=True, type=Path)
    ap.add_argument("--adjudication", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("results"))
    ap.add_argument("--kappa-min", type=float, default=0.60)
    ap.add_argument(
        "--strict-axisb-adjudication", action="store_true",
        help=("require adjudication for Axis-B-only disagreements too. "
              "By default the script reproduces the paper/legacy analysis, "
              "where the adjudication log covers Axis-A/INVALID disagreements "
              "and A1 Axis-B is retained for Axis-B-only disagreements."))
    ap.add_argument("--force", action="store_true",
                    help="write outputs even if validation fails")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(args.key)
    for c in ("blind_id", "model", "question_id"):
        if c not in key.columns:
            sys.exit(f"ERROR: {args.key.name} lacks '{c}'")
    key["category"] = key.question_id.str.extract(r"^C(\d)").astype(int)
    cats = key.set_index("blind_id").category

    a1 = load_labels(args.a1, "A1")
    frames = [a1]
    a2 = None
    if args.a2:
        a2 = load_labels(args.a2, "A2")
        frames.append(a2)

    # 1. Validation
    print("=" * 78)
    print("1. LABEL VALIDATION")
    print("=" * 78)
    problems = []
    for d, who in zip(frames, ["A1", "A2"]):
        problems += validate(d, who, cats)
    unknown = set(a1.blind_id) - set(key.blind_id)
    if unknown:
        problems.append(f"{len(unknown)} blind_ids in A1 are absent from the key")
    for d, who in zip(frames, ["A1", "A2"]):
        print(f"  {who}: {int(d.labelled.sum())}/{len(d)} rows labelled "
              f"({100*d.labelled.mean():.1f}%)  "
              f"ESCALATE={int(d.ESCALATE.sum())}  "
              f"OVER_ABSTENTION={int(d.OVER_ABSTENTION.sum())}")
    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        if not args.force:
            sys.exit("\nValidation failed. Fix the labels, or pass --force to "
                     "write outputs anyway.")
    else:
        print("\n  clean — no guideline violations found")

    # 2. Inter-annotator agreement
    overlap = pd.DataFrame()
    if a2 is not None:
        m = a1[a1.labelled].merge(a2[a2.labelled], on="blind_id",
                                  suffixes=("_1", "_2"))
        m["cat"] = m.blind_id.map(cats)
        m["lab1"] = m.apply(lambda r: "INVALID" if r.INVALID_1 else r.axisA_1, axis=1)
        m["lab2"] = m.apply(lambda r: "INVALID" if r.INVALID_2 else r.axisA_2, axis=1)
        overlap = m
        print("\n" + "=" * 78)
        print(f"2. INTER-ANNOTATOR AGREEMENT  (overlap = {len(m)} rows, "
              f"{100*len(m)/len(key):.1f}% of pool)")
        print("=" * 78)
        if len(m) < 168:
            print(f"  WARNING: overlap is {len(m)} rows; §11 requires >= 168 "
                  f"(10% of 1680)")
        k7, po7, n7 = cohen_kappa(list(m.lab1), list(m.lab2))
        kc, poc, nc = cohen_kappa([collapse(x) for x in m.lab1],
                                  [collapse(y) for y in m.lab2])
        for name, (k, po, n) in (("Axis A (8-class incl. INVALID)", (k7, po7, n7)),
                                 ("Collapsed FAB/ACC/INVALID", (kc, poc, nc))):
            print(f"  {name:32s} kappa={k:.3f} ({interpret(k)})  "
                  f"observed={po:.3f}  N={n}")
        print(f"\n  per-category collapsed kappa:")
        for c in sorted(m.cat.unique()):
            s = m[m.cat == c]
            k, po, n = cohen_kappa([collapse(x) for x in s.lab1],
                                   [collapse(y) for y in s.lab2])
            print(f"    C{c}: kappa={k:.3f}  N={n}" if k is not None
                  else f"    C{c}: n/a  N={n}")
        if kc is not None:
            verdict = "MEETS" if kc >= args.kappa_min else "BELOW"
            print(f"\n  Guideline target (collapsed kappa >= {args.kappa_min}): {verdict}")
            if kc < args.kappa_min:
                print("  §11.5: do NOT adjudicate past this. Find the rule the pair "
                      "is splitting on, amend §8, re-label.")
        # Axis-A/INVALID disagreements are the adjudication set used for the
        # paper's reported results.  We also audit Axis-B differences because
        # confident fabrication depends on assertive vs hedged.  Pass
        # --strict-axisb-adjudication to require those Axis-B-only cases to be
        # present in the adjudication log; the default preserves the legacy
        # paper analysis exactly.
        m["axisA_disagree"] = m.lab1.ne(m.lab2)
        m["axisB_disagree"] = (
            ~m.INVALID_1 & ~m.INVALID_2 & m.axisB_1.ne(m.axisB_2)
        )
        m["needs_adjudication"] = (
            m.axisA_disagree | m.axisB_disagree
            if args.strict_axisb_adjudication else m.axisA_disagree
        )
        m["disagreement_type"] = m.apply(
            lambda r: ("Axis A + Axis B" if r.axisA_disagree and r.axisB_disagree
                       else "Axis A" if r.axisA_disagree else "Axis B"), axis=1)

        dis = m[m.needs_adjudication].copy()
        print(f"\n  Axis-A/INVALID disagreements (paper adjudication set): {int(m.axisA_disagree.sum())}")
        print(f"  Axis-B disagreements (audit):                        {int(m.axisB_disagree.sum())}")
        print(f"  Axis-B-only disagreements (audit):                   {int((m.axisB_disagree & ~m.axisA_disagree).sum())}")
        if args.strict_axisb_adjudication:
            print(f"  strict adjudication cases:                           {len(dis)}")
        else:
            print("  paper-compatible mode: Axis-B-only disagreements do not expand the adjudication set")
        if len(dis):
            top = (m[m.axisA_disagree]
                   .apply(lambda r: " vs ".join(sorted([r.lab1, r.lab2])), axis=1)
                   .value_counts().head(8))
            if len(top):
                print("  most frequent Axis-A label pairs:")
                for pair, cnt in top.items():
                    print(f"    {pair:22s} {cnt}")
            dis[["blind_id", "cat", "disagreement_type", "lab1", "lab2",
                 "axisB_1", "axisB_2"]].to_csv(
                args.outdir / "disagreements.csv", index=False)

    # 3. Adjudication
    adj = pd.DataFrame(columns=["blind_id", "final_label"])
    if args.adjudication and args.adjudication.exists():
        adj = pd.read_csv(args.adjudication)
        print("\n" + "=" * 78)
        print("3. ADJUDICATION LOG")
        print("=" * 78)
        print(f"  rows in log: {len(adj)}")
        if len(overlap):
            dis_ids = set(overlap[overlap.needs_adjudication].blind_id)
            axisA_ids = set(overlap[overlap.axisA_disagree].blind_id)
            axisB_only_ids = set(overlap[overlap.axisB_disagree & ~overlap.axisA_disagree].blind_id)
            log_ids = set(adj.blind_id)
            miss, extra = dis_ids - log_ids, log_ids - dis_ids
            print(f"  adjudication cases expected in this mode: {len(dis_ids)}")
            print(f"    Axis-A/INVALID paper set: {len(axisA_ids)}")
            print(f"    Axis-B-only audit cases:  {len(axisB_only_ids)}")
            if miss:
                print(f"  MISSING from log: {len(miss)} disagreements unresolved")
                missing_axisB_only = miss & axisB_only_ids
                if missing_axisB_only:
                    print(f"    of these, Axis-B-only: {len(missing_axisB_only)}")
                miss_rows = overlap[overlap.blind_id.isin(miss)][
                    ["blind_id", "cat", "disagreement_type", "lab1", "lab2",
                     "axisB_1", "axisB_2"]
                ].merge(key[["blind_id", "question_id", "model"]], on="blind_id", how="left")
                miss_rows.to_csv(args.outdir / "missing_adjudications.csv", index=False)
                print("    -> missing_adjudications.csv")
            if extra:
                print(f"  EXTRA in log: {len(extra)} rows that were not "
                      f"disagreements")
                adj[adj.blind_id.isin(extra)].to_csv(
                    args.outdir / "extra_adjudications.csv", index=False)
                print("    -> extra_adjudications.csv")
            if not miss and not extra:
                print("  consistent — all disagreements required by the selected adjudication mode are covered")

    # 4. Resolve final labels
    long = pd.concat(frames)[["blind_id", "axisA", "axisB", "INVALID",
                              "OVER_ABSTENTION", "ESCALATE", "labelled"]]
    long = long[long.labelled]
    long["lab"] = long.apply(lambda r: "INVALID" if r.INVALID else r.axisA, axis=1)

    first = long.groupby("blind_id").agg(
        lab=("lab", lambda s: s.iloc[0]),
        axisB=("axisB", lambda s: s.iloc[0]),
        over=("OVER_ABSTENTION", "max"),
        esc=("ESCALATE", "max"),
        n_ann=("lab", "size"),
        label_agree=("lab", lambda s: s.nunique() == 1),
        axisB_agree=("axisB", lambda s: s.nunique() == 1),
    ).reset_index()

    adj_map = dict(zip(adj.blind_id, adj.final_label)) if len(adj) else {}
    adjB = dict(zip(adj.blind_id, adj.final_axisB)) if "final_axisB" in adj.columns else {}
    first["final_label"] = [adj_map.get(b, l) for b, l in
                            zip(first.blind_id, first.lab)]
    first["final_axisB"] = [adjB.get(b, x) for b, x in
                            zip(first.blind_id, first.axisB)]

    # Paper-compatible mode requires adjudication for Axis-A/INVALID only.
    # Strict mode additionally requires Axis-B-only disagreements.
    axisb_case = ~first.axisB_agree & first.lab.ne("INVALID")
    first["needs_adjudication"] = (first.n_ann > 1) & (
        (~first.label_agree) | (axisb_case if args.strict_axisb_adjudication else False)
    )

    # Determine whether the available adjudication log resolves the dimensions
    # required by the selected mode.  In paper-compatible mode, A1 Axis-B is
    # retained for Axis-B-only disagreements to reproduce the reported tables.
    adj_label_ids = set(adj.blind_id) if "final_label" in adj.columns else set()
    if "final_axisB" in adj.columns:
        valid_adjB = adj[adj.final_axisB.fillna("").astype(str).str.strip().ne("")]
        adj_axisB_ids = set(valid_adjB.blind_id)
    else:
        adj_axisB_ids = set()

    first["label_unresolved"] = (~first.label_agree) & (~first.blind_id.isin(adj_label_ids))
    first["axisB_unresolved"] = (
        args.strict_axisb_adjudication
        & ~first.axisB_agree & first.final_label.ne("INVALID")
        & ~first.blind_id.isin(adj_axisB_ids)
    )
    first["fully_resolved"] = ~(first.label_unresolved | first.axisB_unresolved)
    unresolved = first[~first.fully_resolved].copy()
    if len(unresolved):
        n_lab = int(unresolved.label_unresolved.sum())
        n_b = int(unresolved.axisB_unresolved.sum())
        print(f"\n  WARNING: {len(unresolved)} disagreements are not fully adjudicated "
              f"(Axis A={n_lab}, Axis B-only={n_b}).")
        print("  A1 values are used only as placeholders for unresolved dimensions; "
              "these rows are NOT final.")
        unresolved.to_csv(args.outdir / "unresolved.csv", index=False)

    first["final_invalid"] = first.final_label.eq("INVALID")
    df = first.merge(key, on="blind_id", how="left")

    # 5. Results
    print("\n" + "=" * 78)
    print("5. RESULTS  (paper convention: NA remains in every rate denominator)")
    print("=" * 78)

    def emit(title, sub, path):
        rows = []
        for model, g in sub.groupby("model"):
            rows.append({"model": model, **rate_block(g)})
        t = pd.DataFrame(rows).sort_values("conf_fab_rate", ascending=False)
        print(f"\n--- {title} ---")
        show = ["model", "n", "gradeable", "NA", "NA_pct",
                "fabrication", "fab_rate", "conf_fab", "conf_fab_rate",
                "conf_fab_ci95"]
        print(t[show].to_string(index=False))
        t.to_csv(path, index=False)
        return t

    emit("Categories 1-3 (source production) — HEADLINE",
         df[df.category.isin([1, 2, 3])], args.outdir / "results_cat123.csv")
    emit("Category 4 (contemporary, reduced rubric) — REPORT SEPARATELY",
         df[df.category == 4], args.outdir / "results_cat4.csv")
    emit("Category 5 (adversarial) — F7 is the target",
         df[df.category == 5], args.outdir / "results_cat5.csv")
    all_t = emit("All categories pooled", df, args.outdir / "results_all.csv")

    # Full final-label distribution used in Appendix C.
    dist = (df.groupby("model").final_label.value_counts().unstack(fill_value=0)
            .reindex(columns=LABELS + ["INVALID"], fill_value=0)
            .rename(columns={"INVALID": "NA"}))
    print("\n--- Final Axis-A distribution (all rows; NA included) ---")
    print(dist.to_string())
    dist.to_csv(args.outdir / "label_distribution.csv")

    # Paper Table 3: fabrication (F1-F4) as a percentage of all responses
    # within each model x category cell.
    cat_rows = []
    for model, gm in df.groupby("model"):
        row = {"model": model}
        for c in sorted(df.category.dropna().unique()):
            gc = gm[gm.category == c]
            row[f"C{int(c)}"] = round(100 * gc.final_label.isin(FAB).sum() / len(gc), 1) if len(gc) else float("nan")
        cat_rows.append(row)
    cat_t = pd.DataFrame(cat_rows)
    print("\n--- Fabrication rate (F1-F4) by category; NA remains denominator ---")
    print(cat_t.to_string(index=False))
    cat_t.to_csv(args.outdir / "fabrication_by_category.csv", index=False)

    # Convenience exports matching the manuscript tables.
    cat123 = []
    for model, g in df[df.category.isin([1, 2, 3])].groupby("model"):
        r = rate_block(g)
        cat123.append({
            "Model": model,
            "Fabrication_pct": r["fab_rate"],
            "Confident_fab_pct": r["conf_fab_rate"],
            "CI95": r["conf_fab_ci95"],
            "NA": r["NA"],
        })
    pd.DataFrame(cat123).sort_values("Confident_fab_pct", ascending=False).to_csv(
        args.outdir / "paper_table2.csv", index=False)

    cat_t.rename(columns={
        "model": "Model", "C1": "C1_Quran", "C2": "C2_Hadith",
        "C3": "C3_Fiqh", "C4": "C4_Contemp", "C5": "C5_Advers"
    }).to_csv(args.outdir / "paper_table3.csv", index=False)

    dist.reset_index().rename(columns={"model": "Model"}).to_csv(
        args.outdir / "paper_table5.csv", index=False)

    # Gradeable sample counts by model and category
    gn = (df.assign(grad=~df.final_invalid)
          .pivot_table(index="model", columns="category", values="grad",
                       aggfunc=["sum", "size"]))
    gn.columns = [f"C{c}_{'gradeable' if a=='sum' else 'n'}" for a, c in gn.columns]
    print("\n--- gradeable n by model x category (§12: never report a rate "
          "without this) ---")
    print(gn.astype(int).to_string())
    gn.astype(int).to_csv(args.outdir / "gradeable_n.csv")

    # Stop-reason audit
    if "stopped" in df.columns:
        print("\n--- Rule 6 audit: INVALID rate by stop reason ---")
        au = (df.assign(inv=df.final_invalid)
              .pivot_table(index="stopped", columns="category", values="inv",
                           aggfunc="mean") * 100).round(1)
        print(au.to_string())
        au.to_csv(args.outdir / "rule6_audit.csv")
        cap = df[df.stopped.isin(["cap", "degenerate-cap"])]
        if len(cap):
            bym = cap.groupby("model").final_invalid.agg(["sum", "size"])
            bym["pct_invalid"] = (100 * bym["sum"] / bym["size"]).round(1)
            print("\n  truncated rows judged INVALID, by model:")
            print(bym.to_string())

    # Flags
    print(f"\n--- flags ---")
    print(f"  OVER_ABSTENTION: {int(df.over.sum())} "
          f"(non-Cat-5: {int(df[df.category != 5].over.sum())})")
    print(f"  ESCALATE (gold disputed): {int(df.esc.sum())}")
    if int(df.esc.sum()):
        df[df.esc][["blind_id", "question_id", "model", "final_label"]].to_csv(
            args.outdir / "escalated.csv", index=False)
        print("    -> escalated.csv; these go to the gold-verification owner, "
              "not to adjudication")

    df.to_csv(args.outdir / "final_labels.csv", index=False)
    print(f"\nWrote outputs to {args.outdir}/")


if __name__ == "__main__":
    main()
