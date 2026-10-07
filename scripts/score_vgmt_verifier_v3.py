#!/usr/bin/env python3
"""Score frozen train inputs with the fixed NLI verifier for VG-MTD V3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


CLAIM_LABELS = ["Supported", "Refuted", "NotEnoughInfo"]
QUERY_LABELS = ["Answer", "Abstain"]
HUMAN_SOURCES = {"human_gold", "human_gold_v2"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(root: Path, path: Path) -> dict:
    resolved = path if path.is_absolute() else root / path
    return json.loads(resolved.read_text(encoding="utf-8"))


def normalize_nli_label(value: object) -> str:
    text = str(value).lower()
    if "entail" in text:
        return "entailment"
    if "contrad" in text:
        return "contradiction"
    if "neutral" in text:
        return "neutral"
    return text


def label_mapping(model: CrossEncoder) -> dict[str, int]:
    raw = getattr(model.model.config, "id2label", {})
    mapping = {
        normalize_nli_label(label): int(index)
        for index, label in raw.items()
        if normalize_nli_label(label)
        in {"entailment", "neutral", "contradiction"}
    }
    required = {"entailment", "neutral", "contradiction"}
    if not required.issubset(mapping):
        return {"contradiction": 0, "entailment": 1, "neutral": 2}
    return mapping


def predict_probabilities(
    model: CrossEncoder,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> np.ndarray:
    scores = np.asarray(
        model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    )
    if scores.ndim != 2 or scores.shape[1] != 3:
        raise ValueError(f"Expected three-way NLI scores, got {scores.shape}")
    looks_like_probs = (
        np.all(scores >= 0)
        and np.allclose(scores.sum(axis=1), 1.0, atol=1e-3)
    )
    if looks_like_probs:
        return scores
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def probabilities_to_row(
    example_id: str,
    task_type: str,
    values: np.ndarray,
    mapping: dict[str, int],
) -> dict:
    p_entail = float(values[mapping["entailment"]])
    p_neutral = float(values[mapping["neutral"]])
    p_contradict = float(values[mapping["contradiction"]])
    claim_values = {
        "Supported": p_entail,
        "Refuted": p_contradict,
        "NotEnoughInfo": p_neutral,
    }
    query_values = {
        "Answer": p_entail,
        "Abstain": max(p_neutral, p_contradict),
    }
    return {
        "source_example_id": example_id,
        "task_type": task_type,
        "p_entail": p_entail,
        "p_neutral": p_neutral,
        "p_contradict": p_contradict,
        "claim_prediction": max(claim_values, key=claim_values.get),
        "query_prediction": max(query_values, key=query_values.get),
    }


def metric_summary(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    confusion = Counter(zip(gold, pred))
    return {
        "rows": len(gold),
        "accuracy": float(accuracy_score(gold, pred)),
        "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro")),
        "gold_distribution": dict(Counter(gold)),
        "prediction_distribution": dict(Counter(pred)),
        "confusion": [
            {"gold": left, "prediction": right, "rows": count}
            for (left, right), count in sorted(confusion.items())
        ],
    }


def rationale_diagnostic(
    model: CrossEncoder,
    outputs: list[dict],
    requests: dict[str, dict],
    mapping: dict[str, int],
    batch_size: int,
) -> dict:
    eligible = []
    for output in outputs:
        request = requests.get(output["request_id"])
        if not request:
            continue
        metadata = request["metadata"]
        if (
            metadata.get("label_source") in HUMAN_SOURCES
            and output["input"]["task_type"] == "claim_verification"
        ):
            eligible.append((output, metadata["reference_label"]))
    pairs = [
        (row["input"]["evidence_text"], row["output"]["rationale"])
        for row, _ in eligible
    ]
    probs = predict_probabilities(model, pairs, batch_size)
    entail = probs[:, mapping["entailment"]]
    correct = np.asarray(
        [row["output"]["decision"] == label for row, label in eligible],
        dtype=bool,
    )
    by_role: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for (row, _), score, is_correct in zip(eligible, entail, correct):
        by_role[row["teacher_role"]].append((float(score), bool(is_correct)))

    result = {
        "included_in_primary_gate": False,
        "reason": (
            "Free-form rationale entailment is not class-conditional and is "
            "inverted for valid insufficiency/abstention explanations."
        ),
        "rows": len(eligible),
        "overall_auc": float(roc_auc_score(correct, entail)),
        "correct_mean_entailment": float(entail[correct].mean()),
        "incorrect_mean_entailment": float(entail[~correct].mean()),
        "by_role": {},
    }
    for role, values in sorted(by_role.items()):
        scores = np.asarray([value[0] for value in values])
        labels = np.asarray([value[1] for value in values], dtype=bool)
        result["by_role"][role] = {
            "rows": len(values),
            "accuracy": float(labels.mean()),
            "correct_mean_entailment": float(scores[labels].mean()),
            "incorrect_mean_entailment": float(scores[~labels].mean()),
            "auc": (
                float(roc_auc_score(labels, scores))
                if len(set(labels.tolist())) > 1
                else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vgmt_supervision_v3.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(root, args.config)
    inputs = config["inputs"]
    verifier = config["verifier"]

    request_rows = read_jsonl(root / inputs["adjudicated_train_requests"])
    requests = {row["request_id"]: row for row in request_rows}
    unique_inputs: dict[str, dict] = {}
    for row in request_rows:
        metadata = row["metadata"]
        if metadata.get("data_split") != "train":
            raise ValueError("Verifier scorer received a non-train request")
        example_id = metadata["source_example_id"]
        unique_inputs.setdefault(example_id, row["input"])
    expected = config["distillation"]["underlying_examples_per_epoch"]
    if len(unique_inputs) != expected:
        raise ValueError(f"Expected {expected} train examples, got {len(unique_inputs)}")

    model_path = root / inputs["nli_model"]
    model = CrossEncoder(
        str(model_path),
        max_length=int(verifier["max_length"]),
        local_files_only=True,
    )
    mapping = label_mapping(model)
    ordered = sorted(unique_inputs.items())
    train_probs = predict_probabilities(
        model,
        [
            (item.get("evidence_text", ""), item.get("claim_text", ""))
            for _, item in ordered
        ],
        int(verifier["batch_size"]),
    )
    score_rows = [
        probabilities_to_row(
            example_id,
            item.get("task_type", ""),
            values,
            mapping,
        )
        for (example_id, item), values in zip(ordered, train_probs)
    ]
    output_path = root / config["outputs"]["verifier_scores"]
    write_jsonl(output_path, score_rows)

    dev = read_jsonl(root / inputs["human_gold_dev"])
    dev_probs = predict_probabilities(
        model,
        [(row["evidence_text"], row["claim_text"]) for row in dev],
        int(verifier["batch_size"]),
    )
    dev_scores = [
        probabilities_to_row(
            row["example_id"],
            row["task_type"],
            values,
            mapping,
        )
        for row, values in zip(dev, dev_probs)
    ]
    claim_pairs = [
        (gold, score)
        for gold, score in zip(dev, dev_scores)
        if gold["task_type"] == "claim_verification"
    ]
    query_pairs = [
        (gold, score)
        for gold, score in zip(dev, dev_scores)
        if gold["task_type"] == "query_decision"
    ]
    report = {
        "config_id": config["config_id"],
        "scope_id": config["scope_id"],
        "model_path": str(model_path.relative_to(root)),
        "model_files": {
            name: sha256(model_path / name)
            for name in ("model.safetensors", "config.json")
        },
        "label_mapping": mapping,
        "train_examples_scored": len(score_rows),
        "test_labels_loaded": False,
        "dev_diagnostics_only": True,
        "human_dev": {
            "claim_verification": metric_summary(
                [gold["gold_label"] for gold, _ in claim_pairs],
                [score["claim_prediction"] for _, score in claim_pairs],
                CLAIM_LABELS,
            ),
            "query_decision": metric_summary(
                [gold["gold_label"] for gold, _ in query_pairs],
                [score["query_prediction"] for _, score in query_pairs],
                QUERY_LABELS,
            ),
        },
    }
    if verifier.get("score_rationale_diagnostic", False):
        outputs = read_jsonl(root / inputs["teacher_outputs"])
        report["rationale_entailment_diagnostic"] = rationale_diagnostic(
            model,
            outputs,
            requests,
            mapping,
            int(verifier["batch_size"]),
        )
    report_path = root / config["outputs"]["verifier_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
