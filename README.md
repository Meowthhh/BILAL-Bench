# BILAL-Bench

BILAL-Bench is a 240-question Bengali benchmark for evaluating fabrication of Islamic sources in open-weight language models. It covers Qur'anic attribution, Hadith attribution, Fiqh rulings, contemporary issues, and adversarial false-premise prompts.

This repository contains the benchmark, annotation files, generation code, and analysis code used to produce the results reported in the paper.

## Repository Structure

```text
BILAL-Bench/
├── data/
│   ├── BILAL_Bench_benchmark.csv
│   └── blind_key.csv
├── annotation/
│   ├── Annotator1_FullDataset_Annotated.csv
│   ├── Annotator2_FullDataset_Annotated.csv
│   ├── adjudication_log.csv
│   └── BILAL_Bench_annotation_guidelines.pdf
├── code/
│   ├── run_benchmark.py
│   ├── run_benchmark_api.py
│   └── analyze_results.py
├── results/
│   ├── paper_table2.csv
│   ├── paper_table3.csv
│   └── paper_table5.csv
├── requirements.txt
└── README.md
```

## Environment

Python 3.10 or 3.11 is recommended. Local model experiments were run on NVIDIA T4 GPUs. Package versions used for reproducibility are specified in `requirements.txt`.

Install the required packages from the pinned environment file:

```bash
pip install -r requirements.txt
```

Gemma checkpoints may require accepting the model terms on Hugging Face and authenticating with:

```bash
hf auth login
```

## Evaluated Models

| Model | Checkpoint / provider | Revision / API model | Prompt |
|---|---|---|---|
| TigerLLM-1B-it | `md-nishat-008/TigerLLM-1B-it` | `bc84e6d` | chat |
| TituLM-gemma-2-2b | `hishab/titulm-gemma-2-2b-v1.1` | `b445fdf` | chat |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | `11c9b30` | chat |
| BanglaLLaMA-3.2-3B-instruct | `BanglaLLM/BanglaLLama-3.2-3b-bangla-alpaca-orca-instruct-v0.0.1` | `8fa0472` | alpaca |
| BanglaLLaMA-3.1-8B-instruct | `BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1` | `ab7305d` | alpaca |
| BanglaLLaMA-3.1-8B-no-pretrain | `BanglaLLM/BanglaLLama-3.1-8b-no-pretrain-bangla-alpaca-orca-instruct-v0.0.1` | `3beb1a1` | alpaca |
| GPT-OSS-120B | Groq | `openai/gpt-oss-120b` | API chat |

Local generation uses deterministic decoding (`do_sample=False`) with an initial generation limit of 512 tokens and a ceiling of 1024 tokens.

## Generate Model Responses

Create an output directory:

```bash
mkdir -p generations
```

Run the six Hugging Face models:

```bash
python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model md-nishat-008/TigerLLM-1B-it --revision bc84e6d --name TigerLLM-1B --prompt-style chat --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/TigerLLM-1B.csv

python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model hishab/titulm-gemma-2-2b-v1.1 --revision b445fdf --name TituLM-gemma-2-2b --prompt-style chat --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/TituLM-gemma-2-2b.csv

python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model google/gemma-2-9b-it --revision 11c9b30 --name Gemma-2-9B --prompt-style chat --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/Gemma-2-9B.csv

python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model BanglaLLM/BanglaLLama-3.2-3b-bangla-alpaca-orca-instruct-v0.0.1 --revision 8fa0472 --name BanglaLLaMA-3.2-3B-inst --prompt-style alpaca --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/BanglaLLaMA-3.2-3B-inst.csv

python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1 --revision ab7305d --name BanglaLLaMA-3.1-8B-inst --prompt-style alpaca --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/BanglaLLaMA-3.1-8B-inst.csv

python code/run_benchmark.py --input data/BILAL_Bench_benchmark.csv --model BanglaLLM/BanglaLLama-3.1-8b-no-pretrain-bangla-alpaca-orca-instruct-v0.0.1 --revision 3beb1a1 --name BanglaLLaMA-3.1-8B-nopre --prompt-style alpaca --condition neutral --max-new-tokens 512 --ceiling 1024 --output generations/BanglaLLaMA-3.1-8B-nopre.csv
```

### GPT-OSS-120B via Groq

Set a Groq API key:

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Then run:

```bash
python code/run_benchmark_api.py \
  --input data/BILAL_Bench_benchmark.csv \
  --model openai/gpt-oss-120b \
  --name GPT-OSS-120B \
  --condition neutral \
  --temperature 0 \
  --reasoning-effort medium \
  --max-new-tokens 512 \
  --ceiling 1024 \
  --output generations/GPT-OSS-120B.csv
```

## Reproduce the Reported Results

The released annotations and adjudication log are used to reproduce the paper statistics and tables:

```bash
python code/analyze_results.py \
  --a1 annotation/Annotator1_FullDataset_Annotated.csv \
  --a2 annotation/Annotator2_FullDataset_Annotated.csv \
  --key data/blind_key.csv \
  --adjudication annotation/adjudication_log.csv \
  --outdir reproduced_results
```

The analysis should reproduce:

```text
Axis-A Cohen's kappa: 0.864
Collapsed Cohen's kappa: 0.893
```

and generate the paper-facing tables:

```text
paper_table2.csv
paper_table3.csv
paper_table5.csv
```

Fabrication and confident-fabrication rates use all responses in the corresponding subset; NA responses remain in the denominator.

## Licenses

The benchmark and code are distributed under the license included with this repository. Third-party models remain subject to their original licenses and terms:

- TigerLLM-1B-it — CC BY 4.0
- TituLM-gemma-2-2b and Gemma-2-9B-it — Gemma terms
- BanglaLLaMA-3.1/3.2 checkpoints — applicable Llama 3.1/3.2 terms
- GPT-OSS-120B — Apache 2.0; hosted inference is subject to Groq API terms

## Citation

Citation information will be added after publication.
