# BILAL-Bench

**Do Open Language Models Invent Scripture? BILAL-Bench: Benchmarking Islamic-Source Fabrication in Bengali**

Sabah Nushra, Md Kishor Morol

Accepted at the 7th Muslims in ML (MusIML) Workshop, NeurIPS 2026.

[Project page](https://meowthhh.github.io/BILAL_Bench_Project/) | [Code](https://github.com/Meowthhh/BILAL-Bench) | [Dataset](https://huggingface.co/datasets/YOUR_USERNAME/BILAL-Bench)

<!-- Replace YOUR_USERNAME/BILAL-Bench above with your Hugging Face dataset repository ID. -->

## About

BILAL-Bench is a 240-question Bengali benchmark for evaluating whether open-weight language models fabricate Islamic sources. The questions cover Qur'anic attribution, Hadith attribution, Fiqh rulings, contemporary issues, and adversarial false-premise prompts.

This repository has the benchmark, the annotation files, the generation code, and the analysis code used for the results in the paper.

## Repository structure

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

## Installation

Python 3.10 or 3.11 is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Evaluated models

| Model | Checkpoint / provider | Revision / API model | Prompt |
|---|---|---|---|
| TigerLLM-1B-it | `md-nishat-008/TigerLLM-1B-it` | `bc84e6d` | chat |
| TituLM-gemma-2-2b | `hishab/titulm-gemma-2-2b-v1.1` | `b445fdf` | chat |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | `11c9b30` | chat |
| BanglaLLaMA-3.2-3B-instruct | `BanglaLLM/BanglaLLama-3.2-3b-bangla-alpaca-orca-instruct-v0.0.1` | `8fa0472` | alpaca |
| BanglaLLaMA-3.1-8B-instruct | `BanglaLLM/BanglaLLama-3.1-8b-bangla-alpaca-orca-instruct-v0.0.1` | `ab7305d` | alpaca |
| BanglaLLaMA-3.1-8B-no-pretrain | `BanglaLLM/BanglaLLama-3.1-8b-no-pretrain-bangla-alpaca-orca-instruct-v0.0.1` | `3beb1a1` | alpaca |
| GPT-OSS-120B | Groq | `openai/gpt-oss-120b` | API chat |

Local models were run on NVIDIA T4 GPUs with greedy decoding (`do_sample=False`), starting at 512 new tokens with a ceiling of 1024.

## Reproducing the results

The released annotations and adjudication log are enough to reproduce the paper's statistics and tables:

```bash
python code/analyze_results.py \
  --a1 annotation/Annotator1_FullDataset_Annotated.csv \
  --a2 annotation/Annotator2_FullDataset_Annotated.csv \
  --key data/blind_key.csv \
  --adjudication annotation/adjudication_log.csv \
  --outdir reproduced_results
```

This should print the inter-annotator agreement (Axis-A Cohen's kappa 0.864, collapsed Cohen's kappa 0.893) and write `paper_table2.csv`, `paper_table3.csv`, and `paper_table5.csv` to `reproduced_results/`. Fabrication and confident-fabrication rates are computed over all responses in each subset, so NA responses stay in the denominator.

## Limitations

BILAL-Bench is an evaluation benchmark, not a source of religious rulings. It uses a single neutral Bengali instruction, and the gold answers for Fiqh items follow the Hanafi position.

## Citation

```bibtex
@inproceedings{nushra2026bilalbench,
  title = {Do Open Language Models Invent Scripture? {BILAL-BENCH}: Benchmarking Islamic-Source Fabrication in Bengali},
  author = {Nushra, Sabah and Morol, Md Kishor},
  booktitle = {The 7th Muslims in ML (MusIML) Workshop at NeurIPS 2026},
  year = {2026}
}
```
