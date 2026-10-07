#!/usr/bin/env python3
"""Summarize the frozen V6 matrix without promoting automatic scores to human truth."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file  # noqa: E402
from scripts.student_v6.evaluate_grounded_predictions_v6 import evaluate  # noqa: E402


HIGHER_IS_BETTER = {
    "answer_accuracy": True,
    "decision_macro_f1": True,
    "claim_support_rate": True,
    "rationale_faithfulness": True,
    "hallucination_rate": False,
    "unsafe_answer_rate": False,
    "abstention_precision": True,
    "abstention_recall": True,
    "abstention_f1": True,
    "evidence_citation_precision": True,
    "response_parse_rate": True,
}


def source_id(row: dict) -> str:
    value = row.get("source_example_id")
    if not value:
        raise ValueError("Grounded row lacks source_example_id")
    return str(value)


def group_id(row: dict) -> str:
    return str(row.get("leakage_group_id") or source_id(row))


def metric_snapshot(rows: list[dict]) -> tuple[dict[str, float], dict]:
    report = evaluate(rows)
    automatic = report["automatic_grounding"]
    flat = {
        "answer_accuracy": float(report["answer_accuracy"]),
        "decision_macro_f1": float(report["decision_macro_f1"]),
        "claim_support_rate": float(automatic["claim_support_rate"]),
        "rationale_faithfulness": float(automatic["rationale_faithfulness"]),
        "hallucination_rate": float(automatic["hallucination_rate"]),
        "unsafe_answer_rate": float(report["unsafe_answer_rate"]),
        "abstention_precision": float(report["abstention_precision"]),
        "abstention_recall": float(report["abstention_recall"]),
        "abstention_f1": float(report["abstention_f1"]),
        "evidence_citation_precision": float(automatic["evidence_citation_precision"]),
        "response_parse_rate": float(automatic["response_parse_rate"]),
    }
    return flat, report


def aggregate_seed_metrics(metrics: dict[int, dict[str, float]]) -> dict:
    result = {}
    for metric in HIGHER_IS_BETTER:
        values = [payload[metric] for _, payload in sorted(metrics.items())]
        result[metric] = {
            "mean": statistics.mean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values_by_seed": {
                str(seed): payload[metric] for seed, payload in sorted(metrics.items())
            },
        }
    return result


def crossed_bootstrap(
    candidate_by_seed: dict[int, list[dict]],
    baseline_by_seed: dict[int, list[dict]],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict:
    if set(candidate_by_seed) != set(baseline_by_seed) or not candidate_by_seed:
        raise ValueError("Candidate and baseline seed sets do not align")
    seeds = sorted(candidate_by_seed)
    candidate_maps = {
        item: {source_id(row): row for row in candidate_by_seed[item]} for item in seeds
    }
    baseline_maps = {
        item: {source_id(row): row for row in baseline_by_seed[item]} for item in seeds
    }
    reference_ids = set(candidate_maps[seeds[0]])
    groups: dict[str, list[str]] = defaultdict(list)
    for training_seed in seeds:
        if set(candidate_maps[training_seed]) != reference_ids:
            raise ValueError("Candidate IDs differ across seeds")
        if set(baseline_maps[training_seed]) != reference_ids:
            raise ValueError("Baseline IDs differ across seeds")
        for item_id in reference_ids:
            left_group = group_id(candidate_maps[training_seed][item_id])
            right_group = group_id(baseline_maps[training_seed][item_id])
            if left_group != right_group:
                raise ValueError("Candidate and baseline leakage groups differ")
            if training_seed == seeds[0]:
                groups[left_group].append(item_id)
    ordered_groups = sorted(groups)
    observed = {}
    for metric in HIGHER_IS_BETTER:
        observed[metric] = float(
            np.mean(
                [
                    metric_snapshot(candidate_by_seed[item])[0][metric]
                    - metric_snapshot(baseline_by_seed[item])[0][metric]
                    for item in seeds
                ]
            )
        )
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in HIGHER_IS_BETTER}
    for _ in range(replicates):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sampled_groups = rng.choice(ordered_groups, size=len(ordered_groups), replace=True)
        by_metric = {metric: [] for metric in HIGHER_IS_BETTER}
        for seed_index, sampled_seed in enumerate(sampled_seeds):
            training_seed = int(sampled_seed)
            candidate_rows = []
            baseline_rows = []
            for group_index, sampled_group in enumerate(sampled_groups):
                for row_index, item_id in enumerate(sorted(groups[str(sampled_group)])):
                    candidate = dict(candidate_maps[training_seed][item_id])
                    baseline = dict(baseline_maps[training_seed][item_id])
                    synthetic_id = f"boot:{seed_index}:{group_index}:{row_index}"
                    synthetic_group = f"boot-group:{group_index}"
                    candidate["source_example_id"] = synthetic_id
                    baseline["source_example_id"] = synthetic_id
                    candidate["leakage_group_id"] = synthetic_group
                    baseline["leakage_group_id"] = synthetic_group
                    candidate_rows.append(candidate)
                    baseline_rows.append(baseline)
            candidate_metrics, _ = metric_snapshot(candidate_rows)
            baseline_metrics, _ = metric_snapshot(baseline_rows)
            for metric in HIGHER_IS_BETTER:
                by_metric[metric].append(candidate_metrics[metric] - baseline_metrics[metric])
        for metric in HIGHER_IS_BETTER:
            draws[metric].append(float(np.mean(by_metric[metric])))
    alpha = 1.0 - confidence
    return {
        metric: {
            "estimate": observed[metric],
            "ci_low": float(np.quantile(values, alpha / 2.0)),
            "ci_high": float(np.quantile(values, 1.0 - alpha / 2.0)),
            "confidence": confidence,
            "replicates": replicates,
            "higher_is_better": HIGHER_IS_BETTER[metric],
        }
        for metric, values in draws.items()
    } | {
        "resampling": {
            "training_seeds": seeds,
            "leakage_groups": len(ordered_groups),
            "method": "crossed_training_seed_by_leakage_group_bootstrap",
        }
    }


def primary_automatic_gates(comparison: dict, reports_by_seed: dict[int, dict], margin: float) -> dict:
    accuracy = comparison["answer_accuracy"]
    macro = comparison["decision_macro_f1"]
    return {
        "decision_quality": (
            (accuracy["ci_low"] > 0.0 or macro["ci_low"] > 0.0)
            and accuracy["estimate"] >= 0.0
            and macro["estimate"] >= 0.0
        ),
        "automatic_claim_support": comparison["claim_support_rate"]["ci_low"] > 0.0,
        "automatic_rationale_faithfulness": comparison["rationale_faithfulness"]["ci_low"] > 0.0,
        "automatic_hallucination": comparison["hallucination_rate"]["ci_high"] < 0.0,
        "unsafe_answer_noninferiority": comparison["unsafe_answer_rate"]["ci_high"] <= margin,
        "abstention_f1_noninferiority": comparison["abstention_f1"]["ci_low"] >= -margin,
        "all_label_recalls_nonzero": all(
            label_metrics["recall"] > 0.0
            for report in reports_by_seed.values()
            for label_metrics in report["by_label"].values()
            if label_metrics["support"] > 0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-manifest", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/vgmt_v6_sealed_results.json"
    )
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/student_training_v6.json"
    )
    parser.add_argument(
        "--method-config", type=Path, default=ROOT / "configs/vgmt_v6.json"
    )
    parser.add_argument(
        "--experiment-matrix", type=Path, default=ROOT / "configs/experiment_matrix.yaml"
    )
    parser.add_argument("--replicates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    method = json.loads(args.method_config.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(args.experiment_matrix.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions_manifest.read_text(encoding="utf-8"))
    if freeze.get("matrix_summarizer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V6 matrix summarizer changed after the freeze")
    if predictions.get("status") != "complete" or predictions.get("sealed_test") is not True:
        raise RuntimeError("V6 grounded prediction matrix is incomplete")
    if predictions.get("freeze_manifest", {}).get("sha256") != sha256_file(args.freeze_manifest):
        raise RuntimeError("Prediction matrix was not produced from this freeze")
    expected = {
        (variant, int(seed))
        for variant in recipe["variants"]
        for seed in recipe["optimization"]["seeds"]
    }
    records = {(row["variant"], int(row["seed"])): row for row in predictions["runs"]}
    if set(records) != expected or len(expected) != 24:
        raise RuntimeError("Prediction manifest does not contain the exact V6 24-run matrix")
    rows_by_run = {}
    metrics_by_variant = {variant: {} for variant in recipe["variants"]}
    reports_by_variant = {variant: {} for variant in recipe["variants"]}
    for key, record in records.items():
        path = Path(record["grounded_output"])
        if not path.exists() or sha256_file(path) != record["grounded_output_sha256"]:
            raise RuntimeError(f"Grounded output missing or changed: {path}")
        rows = read_jsonl(path)
        if len(rows) != int(method["evaluation"]["sealed_test_examples"]):
            raise RuntimeError(f"Unexpected sealed row count: {path}")
        rows_by_run[key] = rows
        flat, report = metric_snapshot(rows)
        metrics_by_variant[key[0]][key[1]] = flat
        reports_by_variant[key[0]][key[1]] = report
    aggregate = {
        variant: aggregate_seed_metrics(metrics_by_variant[variant])
        for variant in recipe["variants"]
    }
    replicates = args.replicates or int(method["evaluation"]["paired_group_bootstrap_replicates"])
    confidence = float(method["evaluation"]["confidence"])
    comparisons = {}
    for index, item in enumerate(matrix["required_comparisons"]):
        candidate, baseline = item["left"], item["right"]
        key = f"{candidate}_minus_{baseline}"
        comparisons[key] = {
            "candidate": candidate,
            "baseline": baseline,
            "primary": bool(item.get("primary", False)),
            "question": item["question"],
            "crossed_bootstrap": crossed_bootstrap(
                {seed: rows_by_run[(candidate, seed)] for seed in recipe["optimization"]["seeds"]},
                {seed: rows_by_run[(baseline, seed)] for seed in recipe["optimization"]["seeds"]},
                replicates=replicates,
                confidence=confidence,
                seed=int(method["seed"]) + index,
            ),
        }
    primary_key = (
        f"{method['evaluation']['primary_method']}_minus_"
        f"{method['evaluation']['primary_baseline']}"
    )
    if primary_key not in comparisons:
        raise RuntimeError("Frozen primary verifier comparison is missing")
    primary = comparisons[primary_key]["crossed_bootstrap"]
    automatic_gates = primary_automatic_gates(
        primary,
        reports_by_variant[method["evaluation"]["primary_method"]],
        float(method["evaluation"]["unsafe_noninferiority_margin"]),
    )
    payload = {
        "status": "complete",
        "contract": "vgmtd_v6_generated_grounding_matrix_summary",
        "result_state": (
            "automatic-gates-pass-human-audit-required"
            if all(automatic_gates.values())
            else "automatic-gates-fail-or-inconclusive"
        ),
        "full_positive_claim_authorized": False,
        "authorization_blocker": (
            "The blinded human claim-support, rationale-faithfulness, and hallucination audit "
            "must be completed before any evidence-grounded positive claim is authorized."
        ),
        "primary_method": method["evaluation"]["primary_method"],
        "primary_baseline": method["evaluation"]["primary_baseline"],
        "role_is_secondary_ablation": True,
        "aggregate_three_seed_metrics": aggregate,
        "comparisons": comparisons,
        "primary_automatic_gates": automatic_gates,
        "automatic_metric_warning": (
            "Cross-encoder grounding scores are diagnostics, not human ground truth."
        ),
        "provenance": {
            "freeze_manifest": {
                "path": str(args.freeze_manifest.resolve()),
                "sha256": sha256_file(args.freeze_manifest),
            },
            "predictions_manifest": {
                "path": str(args.predictions_manifest.resolve()),
                "sha256": sha256_file(args.predictions_manifest),
            },
            "recipe_sha256": sha256_file(args.recipe),
            "method_config_sha256": sha256_file(args.method_config),
            "experiment_matrix_sha256": sha256_file(args.experiment_matrix),
            "summarizer_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
