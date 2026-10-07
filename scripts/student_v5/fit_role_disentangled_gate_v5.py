#!/usr/bin/env python3
"""Fit and audit the train-only role/model-disentangled V5 correctness gate."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .vgmt_v5_contract import (
        CLAIM_LABELS,
        HUMAN_LABEL_SOURCES,
        QUERY_LABELS,
        ROLES,
        TASKS,
        is_valid_teacher_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )
except ImportError:  # direct script execution
    from vgmt_v5_contract import (
        CLAIM_LABELS,
        HUMAN_LABEL_SOURCES,
        QUERY_LABELS,
        ROLES,
        TASKS,
        is_valid_teacher_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )


DECISIONS = CLAIM_LABELS + QUERY_LABELS
VARIANTS = ("full", "no_role", "no_verifier", "no_model", "diagonal_only")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_crossover_gold_index(crossover_outputs: list[dict], expected_sources: int) -> dict[str, dict[str, str]]:
    """Recover the adjudicated train identity independently of legacy provenance.

    The diagonal V2 outputs retain the label source that existed when they were
    generated.  Some of those examples were subsequently human-adjudicated for
    V5, so their old ``label_source`` and ``reference_label`` fields are not the
    V5 gold contract.  The crossover rows are generated only for the frozen V5
    human-train cohort and carry its adjudicated labels; use them as the join
    authority for all nine role--model cells.
    """

    index: dict[str, dict[str, str]] = {}
    for row in crossover_outputs:
        provenance = row.get("provenance", {})
        source_id = provenance.get("source_example_id")
        task_type = row.get("input", {}).get("task_type")
        reference_label = provenance.get("reference_label")
        legal = CLAIM_LABELS if task_type == "claim_verification" else QUERY_LABELS
        if (
            not source_id
            or provenance.get("data_split") != "train"
            or provenance.get("label_source") not in HUMAN_LABEL_SOURCES
            or task_type not in TASKS
            or reference_label not in legal
        ):
            raise ValueError("Crossover row lacks a legal adjudicated V5 train identity")
        identity = {"task_type": task_type, "reference_label": reference_label}
        previous = index.setdefault(source_id, identity)
        if previous != identity:
            raise ValueError(f"Inconsistent crossover gold identity for {source_id}")
    if len(index) != expected_sources:
        raise ValueError("Crossover human-train source count violates the V5 freeze")
    return index


def group_diagonal_by_gold_sources(
    diagonal_outputs: list[dict], gold_index: dict[str, dict[str, str]]
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in diagonal_outputs:
        provenance = row.get("provenance", {})
        source_id = provenance.get("source_example_id")
        if provenance.get("data_split") != "train" or source_id not in gold_index:
            continue
        grouped[source_id].append(row)
    if set(grouped) != set(gold_index):
        raise ValueError("Diagonal outputs do not cover the frozen V5 gold sources")
    return grouped


def clamp(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def verifier_compatibility(score: dict, task: str, decision: str) -> float:
    if task == "query_decision":
        if decision == "Answer":
            return float(score["p_entail"])
        if decision == "Abstain":
            return max(float(score["p_neutral"]), float(score["p_contradict"]))
        return 0.0
    key = {
        "Supported": "p_entail",
        "Refuted": "p_contradict",
        "NotEnoughInfo": "p_neutral",
    }.get(decision)
    return float(score[key]) if key else 0.0


def all_feature_names(models: tuple[str, ...]) -> list[str]:
    names = [
        "verifier_compatibility",
        "diagonal_consensus_fraction",
        "teacher_confidence",
        "schema_valid",
        "task_decision_valid",
        "citation_valid",
    ]
    names.extend(f"task={task}" for task in TASKS)
    names.extend(f"decision={decision}" for decision in DECISIONS)
    names.extend(f"model={model}" for model in models)
    names.extend(f"role={role}" for role in ROLES)
    names.extend(f"task_role={task}|{role}" for task in TASKS for role in ROLES)
    names.extend(f"role_confidence={role}" for role in ROLES)
    names.extend(f"role_verifier={role}" for role in ROLES)
    return names


def feature_values(record: dict, score: dict, models: tuple[str, ...]) -> dict[str, float]:
    task = record["task_type"]
    role = record["teacher_role"]
    model = record["model_name"]
    decision = record["decision"]
    confidence = clamp(record["output"]["output"].get("confidence", 0.0))
    compatibility = verifier_compatibility(score, task, decision)
    validation = record["output"].get("validation", {})
    values: dict[str, float] = {
        "verifier_compatibility": compatibility,
        "diagonal_consensus_fraction": float(record["diagonal_consensus_fraction"]),
        "teacher_confidence": confidence,
        "schema_valid": float(bool(validation.get("schema_valid"))),
        "task_decision_valid": float(bool(validation.get("task_decision_valid"))),
        "citation_valid": float(bool(validation.get("citation_valid"))),
    }
    values.update({f"task={candidate}": float(task == candidate) for candidate in TASKS})
    values.update({f"decision={candidate}": float(decision == candidate) for candidate in DECISIONS})
    values.update({f"model={candidate}": float(model == candidate) for candidate in models})
    values.update({f"role={candidate}": float(role == candidate) for candidate in ROLES})
    values.update(
        {
            f"task_role={candidate_task}|{candidate_role}": float(
                task == candidate_task and role == candidate_role
            )
            for candidate_task in TASKS
            for candidate_role in ROLES
        }
    )
    values.update(
        {f"role_confidence={candidate}": confidence if role == candidate else 0.0 for candidate in ROLES}
    )
    values.update(
        {f"role_verifier={candidate}": compatibility if role == candidate else 0.0 for candidate in ROLES}
    )
    return values


def names_for_variant(names: list[str], variant: str) -> list[str]:
    if variant in {"full", "diagonal_only"}:
        return list(names)
    if variant == "no_role":
        return [name for name in names if not name.startswith(("role=", "task_role=", "role_confidence=", "role_verifier="))]
    if variant == "no_verifier":
        return [name for name in names if name != "verifier_compatibility" and not name.startswith("role_verifier=")]
    if variant == "no_model":
        return [name for name in names if not name.startswith("model=")]
    raise ValueError(f"Unknown gate variant: {variant}")


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    for threshold in np.linspace(0.30, 0.80, 51):
        prediction = probabilities >= threshold
        tp = int(np.sum(prediction & (labels == 1)))
        fp = int(np.sum(prediction & (labels == 0)))
        fn = int(np.sum(~prediction & (labels == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(round(threshold, 2)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "selected_fraction": float(prediction.mean()),
            }
        )
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return float(selected["threshold"]), rows


def fit_variant(records: list[dict], names: list[str], config: dict, seed: int) -> tuple[dict, dict[str, float]]:
    x = np.asarray([[row["features"][name] for name in names] for row in records], dtype=float)
    y = np.asarray([row["correct"] for row in records], dtype=int)
    groups = np.asarray([row["source_example_id"] for row in records])
    if set(y) != {0, 1}:
        raise ValueError("Gate calibration requires both correct and incorrect outputs")
    splitter = StratifiedGroupKFold(
        n_splits=int(config["grouped_folds"]), shuffle=True, random_state=seed
    )
    oof = np.zeros(len(records), dtype=float)
    folds = np.zeros(len(records), dtype=int)
    fold_reports = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), 1):
        model = LogisticRegression(
            C=float(config["c"]),
            class_weight=config["class_weight"],
            solver=config["solver"],
            max_iter=int(config["max_iter"]),
            random_state=seed + fold,
        )
        model.fit(x[train_idx], y[train_idx])
        probability = model.predict_proba(x[valid_idx])[:, 1]
        oof[valid_idx] = probability
        folds[valid_idx] = fold
        fold_reports.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(valid_idx)),
                "validation_groups": int(len(set(groups[valid_idx]))),
                "brier": float(brier_score_loss(y[valid_idx], probability)),
                "log_loss": float(log_loss(y[valid_idx], probability, labels=[0, 1])),
                "auc": float(roc_auc_score(y[valid_idx], probability)),
            }
        )
    if np.any(folds == 0):
        raise AssertionError("OOF prediction missing for one or more records")
    threshold, threshold_sweep = choose_threshold(y, oof)
    final = LogisticRegression(
        C=float(config["c"]),
        class_weight=config["class_weight"],
        solver=config["solver"],
        max_iter=int(config["max_iter"]),
        random_state=seed,
    )
    final.fit(x, y)
    report = {
        "rows": len(records),
        "groups": len(set(groups)),
        "correct_fraction": float(y.mean()),
        "oof_brier": float(brier_score_loss(y, oof)),
        "oof_log_loss": float(log_loss(y, oof, labels=[0, 1])),
        "oof_auc": float(roc_auc_score(y, oof)),
        "oof_average_precision": float(average_precision_score(y, oof)),
        "selected_threshold": threshold,
        "threshold_sweep": threshold_sweep,
        "folds": fold_reports,
        "feature_names": names,
        "classes": [int(value) for value in final.classes_],
        "intercept": [float(value) for value in final.intercept_],
        "coefficients": [float(value) for value in final.coef_[0]],
    }
    predictions = {
        row["request_id"]: {"probability": float(probability), "fold": int(fold)}
        for row, probability, fold in zip(records, oof, folds)
    }
    return report, predictions


def paired_group_bootstrap(
    records: list[dict],
    full_predictions: dict[str, float],
    comparison_predictions: dict[str, float],
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict:
    by_group: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in records:
        request_id = row["request_id"]
        target = float(row["correct"])
        full_error = (full_predictions[request_id] - target) ** 2
        comparison_error = (comparison_predictions[request_id] - target) ** 2
        by_group[row["source_example_id"]].append((full_error, comparison_error))
    group_differences = np.asarray(
        [np.mean([full - comparison for full, comparison in values]) for values in by_group.values()],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(group_differences), size=(replicates, len(group_differences)))
    bootstrap = group_differences[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0])
    estimate = float(group_differences.mean())
    return {
        "estimand": "Brier_full_minus_no_role",
        "estimate": estimate,
        "confidence": confidence,
        "ci_low": float(low),
        "ci_high": float(high),
        "groups": len(group_differences),
        "bootstrap_replicates": replicates,
        "role_value_gate_passed": bool(high < 0.0),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("configs/vgmt_v5.json"))
    parser.add_argument("--verifier-scores", type=Path, default=Path("data/processed/vgmt_v3/verifier_scores_train.jsonl"))
    parser.add_argument("--gate-output", type=Path, default=Path("data/processed/vgmt_v5/role_disentangled_gate_v5.json"))
    parser.add_argument("--oof-output", type=Path, default=Path("data/processed/vgmt_v5/role_disentangled_gate_oof_v5.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("reports/vgmt_role_disentangled_gate_v5.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    cross = config["crossover"]
    gate_config = config["gate"]
    diagonal_path = resolve(root, data["diagonal_outputs"])
    crossover_path = resolve(root, data["crossover_outputs"])
    verifier_path = resolve(root, args.verifier_scores)
    gate_path = resolve(root, args.gate_output)
    oof_path = resolve(root, args.oof_output)
    report_path = resolve(root, args.report)

    models = tuple(cross["models"])
    diagonal_map = cross["diagonal"]
    crossover_outputs = read_jsonl(crossover_path)
    gold_index = build_crossover_gold_index(
        crossover_outputs, int(cross["expected_source_examples"])
    )
    diagonal_by_source = group_diagonal_by_gold_sources(
        read_jsonl(diagonal_path), gold_index
    )

    all_outputs = [row for values in diagonal_by_source.values() for row in values]
    all_outputs.extend(crossover_outputs)
    if len(all_outputs) != int(cross["expected_complete_rows"]):
        raise ValueError(f"Expected {cross['expected_complete_rows']} complete crossover outputs, got {len(all_outputs)}")

    scores = {row["source_example_id"]: row for row in read_jsonl(verifier_path)}
    records: list[dict] = []
    seen_cells: set[tuple[str, str, str]] = set()
    for output in all_outputs:
        provenance = output.get("provenance", {})
        source_id = provenance.get("source_example_id")
        role = output.get("teacher_role")
        model_name = provenance.get("model_name")
        if source_id not in diagonal_by_source or role not in ROLES or model_name not in models:
            raise ValueError("Crossover output escaped the frozen source/role/model domain")
        cell = (source_id, role, model_name)
        if cell in seen_cells:
            raise ValueError(f"Duplicate complete crossover cell: {cell}")
        seen_cells.add(cell)
        if source_id not in scores:
            raise ValueError(f"Missing verifier score for {source_id}")
        decision = output.get("output", {}).get("decision")
        task = output.get("input", {}).get("task_type")
        legal = CLAIM_LABELS if task == "claim_verification" else QUERY_LABELS
        if task not in TASKS or decision not in legal:
            raise ValueError(f"Illegal task/decision in {output.get('request_id')}")
        diagonal_decisions = Counter(row["output"]["decision"] for row in diagonal_by_source[source_id])
        gold_identity = gold_index[source_id]
        if task != gold_identity["task_type"]:
            raise ValueError(f"Task type differs from adjudicated V5 gold for {source_id}")
        reference_label = gold_identity["reference_label"]
        if reference_label not in legal:
            raise ValueError(f"Missing legal human label for {source_id}")
        record = {
            "request_id": output["request_id"],
            "source_example_id": source_id,
            "task_type": task,
            "teacher_role": role,
            "model_name": model_name,
            "decision": decision,
            "correct": int(is_valid_teacher_output(output) and decision == reference_label),
            "is_diagonal": model_name == diagonal_map[role],
            "diagonal_consensus_fraction": diagonal_decisions[decision] / len(diagonal_by_source[source_id]),
            "output": output,
        }
        record["features"] = feature_values(record, scores[source_id], models)
        records.append(record)

    expected_cells = int(cross["expected_complete_rows"])
    if len(seen_cells) != expected_cells:
        raise ValueError("The V5 crossover matrix is not complete")
    all_names = all_feature_names(models)
    seed = int(config["seed"])
    variant_reports: dict[str, dict] = {}
    variant_predictions: dict[str, dict[str, dict]] = {}
    for variant in VARIANTS:
        selected_records = [row for row in records if row["is_diagonal"]] if variant == "diagonal_only" else records
        names = names_for_variant(all_names, variant)
        report, predictions = fit_variant(selected_records, names, gate_config, seed)
        variant_reports[variant] = report
        variant_predictions[variant] = predictions

    full_probability = {key: value["probability"] for key, value in variant_predictions["full"].items()}
    no_role_probability = {key: value["probability"] for key, value in variant_predictions["no_role"].items()}
    role_value = paired_group_bootstrap(
        records,
        full_probability,
        no_role_probability,
        replicates=int(gate_config["bootstrap_replicates"]),
        seed=seed,
        confidence=float(gate_config["role_value_confidence"]),
    )

    oof_rows = []
    for row in records:
        item = {
            "request_id": row["request_id"],
            "source_example_id": row["source_example_id"],
            "task_type": row["task_type"],
            "teacher_role": row["teacher_role"],
            "model_name": row["model_name"],
            "decision": row["decision"],
            "correct": row["correct"],
            "is_diagonal": row["is_diagonal"],
            "probabilities": {},
            "folds": {},
        }
        for variant in VARIANTS:
            prediction = variant_predictions[variant].get(row["request_id"])
            if prediction is not None:
                item["probabilities"][variant] = prediction["probability"]
                item["folds"][variant] = prediction["fold"]
        oof_rows.append(item)
    write_jsonl(oof_path, oof_rows)

    gate_payload = {
        "contract": "role_model_disentangled_correctness_gate_v5",
        "config_id": config["config_id"],
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
        "consensus_definition": "agreement_with_three_frozen_diagonal_outputs",
        "variants": variant_reports,
        "role_value": role_value,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "status": "complete",
        "config_id": config["config_id"],
        "crossover_rows": len(records),
        "source_examples": len({row["source_example_id"] for row in records}),
        "correct_fraction": sum(row["correct"] for row in records) / len(records),
        "role_value": role_value,
        "variant_metrics": {
            variant: {key: value for key, value in payload.items() if key.startswith("oof_") or key in {"rows", "groups", "selected_threshold"}}
            for variant, payload in variant_reports.items()
        },
        "files": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "diagonal_outputs": {"path": str(diagonal_path), "sha256": sha256_file(diagonal_path)},
            "crossover_outputs": {"path": str(crossover_path), "sha256": sha256_file(crossover_path)},
            "verifier_scores": {"path": str(verifier_path), "sha256": sha256_file(verifier_path)},
            "gate": {"path": str(gate_path), "sha256": sha256_file(gate_path)},
            "oof": {"path": str(oof_path), "sha256": sha256_file(oof_path)},
        },
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
