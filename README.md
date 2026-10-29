# Roman Urdu Transformer Baseline

A reproducible comparison of a character n-gram TF-IDF + LinearSVC baseline and `FacebookAI/xlm-roberta-base` for Roman Urdu sentiment classification.

This repository extends earlier work on discourse-aware Roman Urdu sentiment classification:

> Habiba, R., Awais, M., and Shoaib, M. (2020). *A Technique to Calculate National Happiness Index by Analyzing Roman Urdu Messages Posted on Social Media*. ACM Transactions on Asian and Low-Resource Language Information Processing, 19(6), Article 76. https://doi.org/10.1145/3400712

## Public-data policy

This public repository intentionally contains **no research dataset and no experimental outputs**. Place an authorized labelled CSV in `data/` locally. The `.gitignore` file prevents CSV datasets, model checkpoints, and generated outputs from being committed.

Do not publish private tweets, personally identifiable information, API keys, credentials, or restricted research data.

## Experiment

The pipeline:

- removes missing rows and exact duplicate texts before splitting;
- creates a fixed stratified 80/10/10 train/validation/test split;
- trains a character 3–5 gram TF-IDF + class-balanced LinearSVC baseline;
- fine-tunes `FacebookAI/xlm-roberta-base`;
- reports accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class scores, and confusion matrices;
- performs an exact McNemar test on paired test predictions;
- saves prediction logs and a manual linguistic error-analysis template.

No transformer results are claimed in this repository. Run the code on the authorized corpus before reporting any value.

## Dataset format

Create `data/roman_urdu_sentiment.csv` locally:

```csv
text,label
"ye movie bohat achi thi",positive
"service bilkul fazool thi",negative
```

Any label names are accepted. Each class should contain at least ten unique examples for the default split.

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

## Run the experiment

```bash
python run_experiment.py \
  --data data/roman_urdu_sentiment.csv \
  --text-col text \
  --label-col label \
  --output-dir outputs
```

Run only the CPU baseline:

```bash
python run_experiment.py \
  --data data/roman_urdu_sentiment.csv \
  --output-dir outputs_baseline \
  --baseline-only
```

If GPU memory is limited:

```bash
python run_experiment.py \
  --data data/roman_urdu_sentiment.csv \
  --train-batch-size 8 \
  --eval-batch-size 16
```

## Main defaults

| Component | Default |
|---|---|
| Transformer | `FacebookAI/xlm-roberta-base` |
| Split | Stratified 80% train, 10% validation, 10% test |
| Seed | 42 |
| Maximum token length | 128 |
| Epochs | 3 |
| Learning rate | `2e-5` |
| Baseline | Character 3–5 gram TF-IDF + LinearSVC |
| Primary metric | Macro-F1 |

## Outputs

```text
outputs/
├── experiment_config.json
├── data_manifest.json
├── label_mapping.json
├── experiment_summary.md
├── splits/
├── baseline/
├── transformer/
└── comparison/
    ├── test_predictions.csv
    ├── mcnemar_exact.json
    └── manual_error_analysis.csv
```

## Reporting

Use only values produced from the real authorized corpus. Do not compare a new transformer score directly with an older result created from a different split or preprocessing pipeline.

A CV bullet should be completed only after the full run:

> Fine-tuned XLM-RoBERTa-base on **[N] labelled Roman Urdu messages** using a fixed stratified 80/10/10 split; benchmarked it against a character n-gram TF-IDF + LinearSVC baseline, achieving **[accuracy] accuracy and [macro-F1] macro-F1**, with targeted error analysis.

## Limitations

- Exact GPU-level reproducibility may vary across hardware and library builds.
- Random stratified splitting does not prevent user-level, conversation-level, or temporal leakage.
- Aggregate scores should be accompanied by per-class and qualitative error analysis.
- This text-classification experiment does not by itself establish accountable, responsible, or regulation-compliant AI.

## Licence

Code is released under the MIT License. Dataset rights are separate and remain with the respective owners or creators.
