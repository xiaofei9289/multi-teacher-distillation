#!/usr/bin/env python3
"""Frozen selective-reliability metrics and paired group bootstrap for V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from .vgmt_v5_contract import CLAIM_LABELS, QUERY_LABELS, read_jsonl, sha256_file
except ImportError:  # direct script execution
    from vgmt_v5_contract import CLAIM_LABELS, QUERY_LABELS, read_jsonl, sha256_file


ALL_LABELS = CLAIM_LABELS + QUERY_LABELS
FORBIDDEN_TEST_FRAGMENTS = ("human_gold_test", "test_v2", "sealed_test")


def _source_id(row: dict) -> str:
    value = row.get("source_example_id") or row.get("example_id") or row.get("metadata", {}).get("source_example_id")
    if not value:
        raise ValueError("Prediction row lacks a source/example id")
    return str(value)


def _group_id(row: dict) -> str:
    value = row.get("leakage_group_id") or row.get("group_id") or row.get("metadata", {}).get("leakage_group_id")
    return str(value) if value else _source_id(row)


def validate_prediction(row: dict) -> None:
    task = row.get("task_type")
    legal = CLAIM_LABELS if task == "claim_verification" else QUERY_LABELS if task == "query_decision" else ()
    if not legal:
        raise ValueError(f"Unknown task type: {task}")
    if row.get("gold_label") not in legal or row.get("prediction") not in legal:
        raise ValueError("Gold and predicted labels must be task-local")
    probabilities = row.get("decision_probabilities") or row.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(legal):
        raise ValueError("Prediction must contain all and only task-local probabilities")
    values = [float(probabilities[label]) for label in legal]
    if any(not math.isfinite(value) or value < 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-5):
        raise ValueError("Task-local probabilities must form a simplex")


def label_metrics(rows: list[dict]) -> dict:
    metrics = {}
    f1_values = []
    for label in ALL_LABELS:
        relevant = [row for row in rows if label in (CLAIM_LABELS if row["task_type"] == "claim_verification" else QUERY_LABELS)]
        tp = sum(row["gold_label"] == label and row["prediction"] == label for row in relevant)
        fp = sum(row["gold_label"] != label and row["prediction"] == label for row in relevant)
        fn = sum(row["gold_label"] == label and row["prediction"] != label for row in relevant)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support:
            f1_values.append(f1)
    task_macro = {}
    for task, legal in (("claim_verification", CLAIM_LABELS), ("query_decision", QUERY_LABELS)):
        supported = [metrics[label]["f1"] for label in legal if metrics[label]["support"]]
        task_macro[task] = sum(supported) / len(supported) if supported else 0.0
    return {
        "by_label": metrics,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "claim_macro_f1": task_macro["claim_verification"],
        "query_macro_f1": task_macro["query_decision"],
    }


def probability_metrics(rows: list[dict], bins: int = 10) -> dict:
    brier_values = []
    nll_values = []
    confidence_correct = []
    for row in rows:
        legal = CLAIM_LABELS if row["task_type"] == "claim_verification" else QUERY_LABELS
        probabilities = row.get("decision_probabilities") or row["probabilities"]
        brier_values.append(
            sum((float(probabilities[label]) - float(label == row["gold_label"])) ** 2 for label in legal)
        )
        nll_values.append(-math.log(max(float(probabilities[row["gold_label"]]), 1e-12)))
        confidence_correct.append(
            (max(float(value) for value in probabilities.values()), float(row["prediction"] == row["gold_label"]))
        )
    ece = 0.0
    for lower_index in range(bins):
        lower, upper = lower_index / bins, (lower_index + 1) / bins
        members = [item for item in confidence_correct if lower <= item[0] <= upper if lower_index == bins - 1 or item[0] < upper]
        if members:
            ece += len(members) / len(rows) * abs(
                sum(item[1] for item in members) / len(members) - sum(item[0] for item in members) / len(members)
            )
    return {
        "brier": sum(brier_values) / len(brier_values),
        "nll": sum(nll_values) / len(nll_values),
        "ece_10": ece,
    }


def risk_coverage(rows: list[dict]) -> dict:
    scored = []
    for row in rows:
        probabilities = row.get("decision_probabilities") or row["probabilities"]
        confidence = max(float(value) for value in probabilities.values())
        scored.append((confidence, _source_id(row), int(row["prediction"] != row["gold_label"])))
    scored.sort(key=lambda item: (-item[0], item[1]))
    errors = 0
    curve = []
    for index, (confidence, _, error) in enumerate(scored, 1):
        errors += error
        curve.append({"coverage": index / len(scored), "risk": errors / index, "threshold": confidence})
    aurc = sum(point["risk"] for point in curve) / len(curve) if curve else 0.0
    return {"aurc": aurc, "curve": curve}


def evaluate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Evaluation rows are empty")
    ids = [_source_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation contains duplicate source/example ids")
    for row in rows:
        validate_prediction(row)
    labels = label_metrics(rows)
    claim_safe = [row for row in rows if row["task_type"] == "claim_verification" and row["gold_label"] == "NotEnoughInfo"]
    query_safe = [row for row in rows if row["task_type"] == "query_decision" and row["gold_label"] == "Abstain"]
    claim_unsafe = sum(row["prediction"] in {"Supported", "Refuted"} for row in claim_safe)
    query_unsafe = sum(row["prediction"] == "Answer" for row in query_safe)
    selective = risk_coverage(rows)
    probability = probability_metrics(rows)
    positive_actions = {"Supported", "Refuted", "Answer"}
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    generated_tokens = [int(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None]
    return {
        "examples": len(rows),
        "accuracy": sum(row["prediction"] == row["gold_label"] for row in rows) / len(rows),
        "macro_f1": labels["macro_f1"],
        "claim_macro_f1": labels["claim_macro_f1"],
        "query_macro_f1": labels["query_macro_f1"],
        "by_label": labels["by_label"],
        "claim_unsafe_error": claim_unsafe / len(claim_safe) if claim_safe else 0.0,
        "claim_unsafe_numerator": claim_unsafe,
        "claim_unsafe_denominator": len(claim_safe),
        "query_unsafe_error": query_unsafe / len(query_safe) if query_safe else 0.0,
        "query_unsafe_numerator": query_unsafe,
        "query_unsafe_denominator": len(query_safe),
        "aurc": selective["aurc"],
        "coverage": sum(row["prediction"] in positive_actions for row in rows) / len(rows),
        **probability,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "mean_generated_tokens": sum(generated_tokens) / len(generated_tokens) if generated_tokens else None,
        "risk_coverage_curve": selective["curve"],
        "prediction_counts": dict(sorted(Counter(row["prediction"] for row in rows).items())),
    }


def _metric_difference(candidate: list[dict], baseline: list[dict], metric: str) -> float:
    candidate_metrics = evaluate(candidate)
    baseline_metrics = evaluate(baseline)
    return float(candidate_metrics[metric] - baseline_metrics[metric])


def paired_group_bootstrap(candidate: list[dict], baseline: list[dict], *, replicates: int, confidence: float, seed: int) -> dict:
    candidate_by_id = {_source_id(row): row for row in candidate}
    baseline_by_id = {_source_id(row): row for row in baseline}
    if set(candidate_by_id) != set(baseline_by_id):
        raise ValueError("Candidate and baseline prediction ids do not align")
    for source_id in candidate_by_id:
        if _group_id(candidate_by_id[source_id]) != _group_id(baseline_by_id[source_id]):
            raise ValueError(f"Candidate and baseline group ids do not align for {source_id}")
    group_to_ids: dict[str, list[str]] = {}
    for source_id, row in candidate_by_id.items():
        group_to_ids.setdefault(_group_id(row), []).append(source_id)
    groups = sorted(group_to_ids)
    observed = {
        metric: _metric_difference(candidate, baseline, metric)
        for metric in ("macro_f1", "claim_unsafe_error", "query_unsafe_error", "aurc")
    }
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in observed}
    for _ in range(replicates):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        candidate_sample = []
        baseline_sample = []
        for group_index, group_id in enumerate(sampled):
            for row_index, source_id in enumerate(sorted(group_to_ids[str(group_id)])):
                candidate_row = dict(candidate_by_id[source_id])
                baseline_row = dict(baseline_by_id[source_id])
                synthetic_id = f"boot:{group_index}:{row_index}"
                synthetic_group = f"boot-group:{group_index}"
                candidate_row["source_example_id"] = synthetic_id
                baseline_row["source_example_id"] = synthetic_id
                candidate_row["leakage_group_id"] = synthetic_group
                baseline_row["leakage_group_id"] = synthetic_group
                candidate_sample.append(candidate_row)
                baseline_sample.append(baseline_row)
        candidate_metrics = evaluate(candidate_sample)
        baseline_metrics = evaluate(baseline_sample)
        for metric in draws:
            draws[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
    alpha = 1.0 - confidence
    return {
        metric: {
            "estimate": observed[metric],
            "ci_low": float(np.quantile(values, alpha / 2.0)),
            "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "two_sided_p": min(
                1.0,
                2.0 * min(
                    (1 + sum(value <= 0.0 for value in values)) / (len(values) + 1),
                    (1 + sum(value >= 0.0 for value in values)) / (len(values) + 1),
                ),
            ),
            "confidence": confidence,
            "replicates": replicates,
        }
        for metric, values in draws.items()
    }


def multi_seed_group_bootstrap(
    candidate_by_seed: dict[int, list[dict]],
    baseline_by_seed: dict[int, list[dict]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict:
    """Crossed bootstrap over training seeds and source/leakage groups."""
    if set(candidate_by_seed) != set(baseline_by_seed) or not candidate_by_seed:
        raise ValueError("Candidate and baseline seed sets do not align")
    seeds = sorted(candidate_by_seed)
    candidate_maps = {item: {_source_id(row): row for row in candidate_by_seed[item]} for item in seeds}
    baseline_maps = {item: {_source_id(row): row for row in baseline_by_seed[item]} for item in seeds}
    reference_ids = set(candidate_maps[seeds[0]])
    group_to_ids: dict[str, list[str]] = {}
    for training_seed in seeds:
        if set(candidate_maps[training_seed]) != reference_ids or set(baseline_maps[training_seed]) != reference_ids:
            raise ValueError("Prediction ids do not align across systems and training seeds")
        for source_id in reference_ids:
            candidate_group = _group_id(candidate_maps[training_seed][source_id])
            baseline_group = _group_id(baseline_maps[training_seed][source_id])
            if candidate_group != baseline_group:
                raise ValueError(f"Group ids do not align for seed={training_seed}, source={source_id}")
            if training_seed == seeds[0]:
                group_to_ids.setdefault(candidate_group, []).append(source_id)
            elif source_id not in group_to_ids.get(candidate_group, []):
                raise ValueError("Leakage-group assignments differ across training seeds")
    groups = sorted(group_to_ids)
    metrics = ("macro_f1", "claim_unsafe_error", "query_unsafe_error", "aurc")
    observed = {
        metric: float(
            np.mean(
                [
                    evaluate(candidate_by_seed[training_seed])[metric]
                    - evaluate(baseline_by_seed[training_seed])[metric]
                    for training_seed in seeds
                ]
            )
        )
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        replicate_differences = {metric: [] for metric in metrics}
        for seed_index, sampled_seed in enumerate(sampled_seeds):
            training_seed = int(sampled_seed)
            candidate_sample = []
            baseline_sample = []
            for group_index, group_id in enumerate(sampled_groups):
                for row_index, source_id in enumerate(sorted(group_to_ids[str(group_id)])):
                    candidate_row = dict(candidate_maps[training_seed][source_id])
                    baseline_row = dict(baseline_maps[training_seed][source_id])
                    synthetic_id = f"boot:{seed_index}:{group_index}:{row_index}"
                    synthetic_group = f"boot-group:{group_index}"
                    candidate_row["source_example_id"] = synthetic_id
                    baseline_row["source_example_id"] = synthetic_id
                    candidate_row["leakage_group_id"] = synthetic_group
                    baseline_row["leakage_group_id"] = synthetic_group
                    candidate_sample.append(candidate_row)
                    baseline_sample.append(baseline_row)
            candidate_metrics = evaluate(candidate_sample)
            baseline_metrics = evaluate(baseline_sample)
            for metric in metrics:
                replicate_differences[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
        for metric in metrics:
            draws[metric].append(float(np.mean(replicate_differences[metric])))
    alpha = 1.0 - confidence
    return {
        metric: {
            "estimate": observed[metric],
            "ci_low": float(np.quantile(values, alpha / 2.0)),
            "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "two_sided_p": min(
                1.0,
                2.0 * min(
                    (1 + sum(value <= 0.0 for value in values)) / (len(values) + 1),
                    (1 + sum(value >= 0.0 for value in values)) / (len(values) + 1),
                ),
            ),
            "confidence": confidence,
            "replicates": replicates,
        }
        for metric, values in draws.items()
    } | {
        "resampling": {
            "training_seeds": seeds,
            "seed_draws_per_replicate": len(seeds),
            "leakage_groups": len(groups),
            "group_draws_per_replicate": len(groups),
        }
    }


def verify_sealed_authorization(paths: list[Path], freeze_manifest: Path | None, sealed_test: bool) -> dict | None:
    is_test = any(any(fragment in path.as_posix().casefold() for fragment in FORBIDDEN_TEST_FRAGMENTS) for path in paths)
    if not is_test and not sealed_test:
        return None
    if not sealed_test or freeze_manifest is None:
        raise RuntimeError("Sealed-test paths require --sealed-test and --freeze-manifest")
    manifest = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("test_open_authorized") is not True:
        raise RuntimeError("Freeze manifest does not authorize the one-shot sealed evaluation")
    expected_evaluator = manifest.get("evaluator_sha256")
    current_evaluator = sha256_file(Path(__file__))
    if expected_evaluator != current_evaluator:
        raise RuntimeError("Evaluator hash differs from the frozen manifest")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/vgmt_v5.json"))
    parser.add_argument("--gate-report", type=Path, default=Path("reports/vgmt_role_disentangled_gate_v5.json"))
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    resolve = lambda path: path if path.is_absolute() else root / path
    candidate_path, baseline_path = resolve(args.candidate), resolve(args.baseline)
    config_path, gate_report_path = resolve(args.config), resolve(args.gate_report)
    freeze_path = resolve(args.freeze_manifest) if args.freeze_manifest else None
    freeze = verify_sealed_authorization([candidate_path, baseline_path], freeze_path, args.sealed_test)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate, baseline = read_jsonl(candidate_path), read_jsonl(baseline_path)
    candidate_metrics, baseline_metrics = evaluate(candidate), evaluate(baseline)
    evaluation = config["evaluation"]
    comparison = paired_group_bootstrap(
        candidate,
        baseline,
        replicates=args.replicates or int(evaluation["paired_group_bootstrap_replicates"]),
        confidence=float(evaluation["confidence"]),
        seed=int(config["seed"]),
    )
    role_gate = None
    if gate_report_path.exists():
        role_gate = json.loads(gate_report_path.read_text(encoding="utf-8"))["role_value"]["role_value_gate_passed"]
    margin = float(evaluation["unsafe_noninferiority_margin"])
    gates = {
        "macro_f1_superiority": comparison["macro_f1"]["ci_low"] > 0.0,
        "claim_unsafe_noninferiority": comparison["claim_unsafe_error"]["ci_high"] <= margin,
        "query_unsafe_noninferiority": comparison["query_unsafe_error"]["ci_high"] <= margin,
        "all_label_recalls_nonzero": all(value["recall"] > 0.0 for value in candidate_metrics["by_label"].values()),
        "aurc_improved": comparison["aurc"]["ci_high"] < 0.0,
        "gate_level_role_value": role_gate is True,
    }
    report = {
        "contract": "vgmtd_v5_sealed_selective_evaluation",
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "paired_group_bootstrap": comparison,
        "pairwise_diagnostic_gates": gates,
        "pairwise_diagnostic_passed": all(gates.values()),
        "full_positive_claim_authorized": False,
        "authorization_note": "Only summarize_formal_matrix_v5.py can authorize the V5 claim after all 24 frozen runs.",
        "sealed_test": bool(freeze),
        "freeze_manifest": str(freeze_path) if freeze_path else None,
        "files": {
            "candidate": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "baseline": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "evaluator": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        },
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
