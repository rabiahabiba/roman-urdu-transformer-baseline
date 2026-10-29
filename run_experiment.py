#!/usr/bin/env python3
"""Reproducible Roman Urdu sentiment baseline and transformer experiment.

Compares character n-gram TF-IDF + LinearSVC with
FacebookAI/xlm-roberta-base on the same fixed stratified test set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
USER_PATTERN = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TF-IDF + LinearSVC with XLM-RoBERTa-base."
    )
    parser.add_argument("--data", required=True, help="Labelled CSV file.")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--error-sample-size", type=int, default=100)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the classical CPU baseline.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clean_text(value: Any) -> str:
    """Minimal cleaning that preserves punctuation, hashtags, and emojis."""
    text = str(value).strip()
    text = URL_PATTERN.sub(" <URL> ", text)
    text = USER_PATTERN.sub(" <USER> ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(args.data)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing = [
        column
        for column in (args.text_col, args.label_col)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    raw_rows = len(frame)
    frame = frame[[args.text_col, args.label_col]].copy()
    frame.columns = ["text", "label"]
    frame = frame.dropna(subset=["text", "label"])
    after_missing = len(frame)

    frame["text"] = frame["text"].map(clean_text)
    frame["label"] = frame["label"].astype(str).str.strip()
    frame = frame[(frame["text"] != "") & (frame["label"] != "")]
    after_empty = len(frame)

    frame = frame.drop_duplicates(subset=["text"], keep="first").reset_index(
        drop=True
    )
    counts = frame["label"].value_counts().sort_index()
    if len(counts) < 2:
        raise ValueError("At least two sentiment classes are required.")
    if counts.min() < 10:
        raise ValueError(
            "Each class needs at least 10 unique examples for the default "
            f"80/10/10 split. Counts: {counts.to_dict()}"
        )

    manifest = {
        "source_file": path.name,
        "raw_rows": raw_rows,
        "rows_after_dropping_missing": after_missing,
        "rows_after_dropping_empty": after_empty,
        "unique_rows_after_text_deduplication": len(frame),
        "duplicates_removed": after_empty - len(frame),
        "class_distribution": {
            str(label): int(count) for label, count in counts.items()
        },
        "cleaning": [
            "trim whitespace",
            "replace URLs with <URL>",
            "replace @mentions with <USER>",
            "collapse repeated whitespace",
            "preserve punctuation, hashtags, and emojis",
        ],
    }
    return frame, manifest


def split_data(
    frame: pd.DataFrame,
    validation_size: float,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("Validation and test sizes must be greater than zero.")
    if validation_size + test_size >= 1:
        raise ValueError("Validation size plus test size must be below one.")

    holdout_size = validation_size + test_size
    train, holdout = train_test_split(
        frame,
        test_size=holdout_size,
        random_state=seed,
        stratify=frame["label_id"],
    )
    validation, test = train_test_split(
        holdout,
        test_size=test_size / holdout_size,
        random_state=seed,
        stratify=holdout["label_id"],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def save_evaluation(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = metric_values(y_true, y_pred)
    write_json(output_dir / "metrics.json", metrics)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        output_dir / "classification_report.csv",
        index=True,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    pd.DataFrame(
        matrix,
        index=[f"true_{name}" for name in class_names],
        columns=[f"pred_{name}" for name in class_names],
    ).to_csv(output_dir / "confusion_matrix.csv", index=True)
    return metrics


def train_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    class_names: list[str],
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    lowercase=True,
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(class_weight="balanced", random_state=seed),
            ),
        ]
    )
    pipeline.fit(train["text"], train["label_id"])
    predictions = pipeline.predict(test["text"]).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "tfidf_linearsvc.joblib")
    metrics = save_evaluation(
        output_dir,
        test["label_id"].to_numpy(),
        predictions,
        class_names,
    )
    return predictions, metrics


def train_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    class_names: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer dependencies are missing. Install requirements.txt "
            "or use --baseline-only."
        ) from exc

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def convert(frame: pd.DataFrame) -> Any:
        subset = frame[["text", "label_id"]].rename(
            columns={"label_id": "labels"}
        )
        dataset = Dataset.from_pandas(subset, preserve_index=False)
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=args.max_length,
            ),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing",
        )

    train_dataset = convert(train)
    validation_dataset = convert(validation)
    test_dataset = convert(test)

    id2label = {index: name for index, name in enumerate(class_names)}
    label2id = {name: index for index, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_trainer_metrics(eval_prediction: Any) -> dict[str, float]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        return metric_values(eval_prediction.label_ids, predictions)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.10,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_trainer_metrics,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    prediction_output = trainer.predict(test_dataset)
    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1).astype(int)

    metrics = save_evaluation(
        output_dir,
        test["label_id"].to_numpy(),
        predictions,
        class_names,
    )
    return predictions, metrics


def exact_mcnemar(
    y_true: np.ndarray,
    baseline_predictions: np.ndarray,
    transformer_predictions: np.ndarray,
) -> dict[str, Any]:
    baseline_correct = baseline_predictions == y_true
    transformer_correct = transformer_predictions == y_true

    baseline_only = int(
        np.sum(baseline_correct & ~transformer_correct)
    )
    transformer_only = int(
        np.sum(~baseline_correct & transformer_correct)
    )
    discordant = baseline_only + transformer_only
    p_value = (
        float(
            binomtest(
                baseline_only,
                n=discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
        if discordant
        else 1.0
    )
    return {
        "test": "Exact McNemar test using a two-sided binomial test",
        "baseline_correct_transformer_wrong": baseline_only,
        "baseline_wrong_transformer_correct": transformer_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
        "alpha": 0.05,
        "statistically_significant_at_0.05": bool(p_value < 0.05),
    }


def prediction_log(
    test: pd.DataFrame,
    class_names: list[str],
    baseline_predictions: np.ndarray,
    transformer_predictions: np.ndarray | None,
) -> pd.DataFrame:
    result = test[["text", "label", "label_id"]].copy()
    result["baseline_prediction"] = [
        class_names[value] for value in baseline_predictions
    ]
    result["baseline_correct"] = (
        baseline_predictions == test["label_id"].to_numpy()
    )

    if transformer_predictions is not None:
        result["transformer_prediction"] = [
            class_names[value] for value in transformer_predictions
        ]
        result["transformer_correct"] = (
            transformer_predictions == test["label_id"].to_numpy()
        )
        result["models_disagree"] = (
            baseline_predictions != transformer_predictions
        )
    return result


def error_analysis_template(
    log: pd.DataFrame,
    path: Path,
    sample_size: int,
    seed: int,
) -> None:
    if "transformer_correct" not in log.columns:
        return

    errors = log.loc[~log["transformer_correct"]].copy()
    if len(errors) > sample_size:
        errors = errors.sample(n=sample_size, random_state=seed)

    for column in (
        "spelling_variation",
        "code_mixing",
        "informal_expression",
        "negation",
        "multiple_subopinions",
        "annotation_notes",
    ):
        errors[column] = ""
    errors.to_csv(path, index=False)


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    train_size: int,
    validation_size: int,
    test_size: int,
    baseline_metrics: dict[str, float],
    transformer_metrics: dict[str, float] | None,
    mcnemar: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Experiment summary",
        "",
        "## Dataset",
        "",
        f"- Unique messages: {manifest['unique_rows_after_text_deduplication']}",
        f"- Class distribution: {json.dumps(manifest['class_distribution'], sort_keys=True)}",
        (
            f"- Split: train={train_size}, validation={validation_size}, "
            f"test={test_size}"
        ),
        f"- Random seed: {args.seed}",
        "",
        "## Results",
        "",
        "| Model | Accuracy | Macro-F1 | Weighted-F1 |",
        "|---|---:|---:|---:|",
        (
            "| Character TF-IDF + LinearSVC | "
            f"{baseline_metrics['accuracy']:.4f} | "
            f"{baseline_metrics['macro_f1']:.4f} | "
            f"{baseline_metrics['weighted_f1']:.4f} |"
        ),
    ]

    if transformer_metrics is None:
        lines.append("| Transformer | Not run | Not run | Not run |")
    else:
        lines.append(
            f"| {args.model_name} | "
            f"{transformer_metrics['accuracy']:.4f} | "
            f"{transformer_metrics['macro_f1']:.4f} | "
            f"{transformer_metrics['weighted_f1']:.4f} |"
        )

    if mcnemar is not None:
        lines.extend(
            [
                "",
                "## Paired significance test",
                "",
                f"- Exact McNemar p-value: {mcnemar['p_value']:.6g}",
                (
                    "- Significant at alpha=0.05: "
                    f"{mcnemar['statistically_significant_at_0.05']}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Reporting note",
            "",
            (
                "These results apply only to the recorded dataset, cleaning "
                "steps, fixed split, and configuration. Do not compare them "
                "directly with scores from another split."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "experiment_config.json", vars(args))

    frame, manifest = load_data(args)
    encoder = LabelEncoder()
    frame["label_id"] = encoder.fit_transform(frame["label"])
    class_names = [str(name) for name in encoder.classes_]

    write_json(output_root / "data_manifest.json", manifest)
    write_json(
        output_root / "label_mapping.json",
        {str(index): name for index, name in enumerate(class_names)},
    )

    train, validation, test = split_data(
        frame,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    split_dir = output_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / "train.csv", index=False)
    validation.to_csv(split_dir / "validation.csv", index=False)
    test.to_csv(split_dir / "test.csv", index=False)

    baseline_predictions, baseline_metrics = train_baseline(
        train,
        test,
        output_root / "baseline",
        class_names,
        args.seed,
    )

    transformer_predictions = None
    transformer_metrics = None
    mcnemar = None

    if not args.baseline_only:
        transformer_predictions, transformer_metrics = train_transformer(
            train,
            validation,
            test,
            output_root / "transformer",
            args,
            class_names,
        )
        mcnemar = exact_mcnemar(
            test["label_id"].to_numpy(),
            baseline_predictions,
            transformer_predictions,
        )
        write_json(
            output_root / "comparison" / "mcnemar_exact.json",
            mcnemar,
        )

    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    log = prediction_log(
        test,
        class_names,
        baseline_predictions,
        transformer_predictions,
    )
    log.to_csv(comparison_dir / "test_predictions.csv", index=False)
    error_analysis_template(
        log,
        comparison_dir / "manual_error_analysis.csv",
        args.error_sample_size,
        args.seed,
    )

    write_summary(
        output_root / "experiment_summary.md",
        manifest,
        len(train),
        len(validation),
        len(test),
        baseline_metrics,
        transformer_metrics,
        mcnemar,
        args,
    )
    print(f"Experiment completed. Results: {output_root.resolve()}")


if __name__ == "__main__":
    main()
