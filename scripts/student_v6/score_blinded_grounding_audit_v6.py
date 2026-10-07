#!/usr/bin/env python3
"""Validate, adjudicate, and score the frozen V6 blinded human audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file  # noqa: E402
from scripts.student_v6.build_blinded_grounding_audit_v6 import (  # noqa: E402
    BLIND_COLUMNS,
    POSITIVE,
)


JUDGMENT_FIELDS = (
    "answer_supported",
    "rationale_faithful",
    "unsupported_material_claim",
    "citation_valid",
)
ALLOWED = frozenset({"Yes", "No", "Unclear"})
IMMUTABLE_COLUMNS = tuple(
    column
    for column in BLIND_COLUMNS
    if column not in JUDGMENT_FIELDS and column not in {"annotator_id", "notes"}
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def indexed(rows: list[dict[str, str]], *, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = row.get("audit_id", "").strip()
        if not audit_id or audit_id in result:
            raise ValueError(f"{label} contains a missing or duplicate audit_id")
        result[audit_id] = row
    return result


def validate_completed_sheet(
    path: Path,
    base: dict[str, dict[str, str]],
    *,
    label: str,
) -> tuple[dict[str, dict[str, str]], str]:
    columns, rows = read_csv(path)
    if tuple(columns) != BLIND_COLUMNS:
        raise ValueError(f"{label} columns differ from the frozen blind worksheet")
    completed = indexed(rows, label=label)
    if set(completed) != set(base):
        raise ValueError(f"{label} audit IDs differ from the frozen blind worksheet")
    annotators = set()
    for audit_id, row in completed.items():
        source = base[audit_id]
        for column in IMMUTABLE_COLUMNS:
            if row.get(column, "") != source.get(column, ""):
                raise ValueError(f"{label} changed blinded content: {audit_id}:{column}")
        for field in JUDGMENT_FIELDS:
            if row.get(field, "").strip() not in ALLOWED:
                raise ValueError(f"{label} has invalid judgment: {audit_id}:{field}")
        annotator = row.get("annotator_id", "").strip()
        if not annotator:
            raise ValueError(f"{label} lacks annotator_id: {audit_id}")
        annotators.add(annotator)
    if len(annotators) != 1:
        raise ValueError(f"{label} must contain exactly one stable annotator_id")
    return completed, next(iter(annotators))


def cohen_kappa(left: list[str], right: list[str]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires two non-empty aligned judgment lists")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in ALLOWED
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def resolve_judgments(
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    adjudicated: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    final: dict[str, dict[str, str]] = {}
    disagreements = {field: 0 for field in JUDGMENT_FIELDS}
    for audit_id in sorted(first):
        final[audit_id] = {}
        for field in JUDGMENT_FIELDS:
            left = first[audit_id][field].strip()
            right = second[audit_id][field].strip()
            if left == right:
                final[audit_id][field] = left
            else:
                disagreements[field] += 1
                final[audit_id][field] = adjudicated[audit_id][field].strip()
    return final, disagreements


def human_values(row: dict) -> dict[str, float]:
    judgment = row["judgment"]
    return {
        "claim_support_rate": float(
            judgment["answer_supported"] == "Yes"
            and judgment["citation_valid"] == "Yes"
        ),
        "rationale_faithfulness": float(judgment["rationale_faithful"] == "Yes"),
        "hallucination_rate": float(
            judgment["unsupported_material_claim"] != "No"
            or judgment["citation_valid"] != "Yes"
        ),
        "citation_valid_rate": float(judgment["citation_valid"] == "Yes"),
    }


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        raise ValueError("Human grounding metrics require positive-output rows")
    values = [human_values(row) for row in rows]
    return {
        metric: float(np.mean([item[metric] for item in values]))
        for metric in values[0]
    }


def bootstrap_difference(
    candidate: list[dict],
    baseline: list[dict],
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict:
    if not candidate or not baseline or replicates <= 0:
        raise ValueError("Human bootstrap requires non-empty samples and replicates")
    candidate_values = [human_values(row) for row in candidate]
    baseline_values = [human_values(row) for row in baseline]
    metrics = tuple(candidate_values[0])
    observed = {
        metric: float(
            np.mean([row[metric] for row in candidate_values])
            - np.mean([row[metric] for row in baseline_values])
        )
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(replicates):
        candidate_indices = rng.integers(0, len(candidate_values), len(candidate_values))
        baseline_indices = rng.integers(0, len(baseline_values), len(baseline_values))
        for metric in metrics:
            draws[metric].append(
                float(
                    np.mean([candidate_values[index][metric] for index in candidate_indices])
                    - np.mean([baseline_values[index][metric] for index in baseline_indices])
                )
            )
    alpha = 1.0 - confidence
    return {
        metric: {
            "estimate": observed[metric],
            "ci_low": float(np.quantile(draws[metric], alpha / 2.0)),
            "ci_high": float(np.quantile(draws[metric], 1.0 - alpha / 2.0)),
            "confidence": confidence,
            "replicates": replicates,
            "higher_is_better": metric != "hallucination_rate",
        }
        for metric in metrics
    } | {
        "resampling": {
            "method": "independent_stratified_bootstrap_over_blinded_positive_outputs",
            "candidate_rows": len(candidate),
            "baseline_rows": len(baseline),
            "reference_training_seed_only": True,
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-csv", type=Path, required=True)
    parser.add_argument("--annotator-one", type=Path, required=True)
    parser.add_argument("--annotator-two", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--hidden-key", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--automatic-results", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/human_grounding_audit_v6.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/vgmt_v6_human_grounding_results.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    automatic = json.loads(args.automatic_results.read_text(encoding="utf-8"))
    if freeze.get("human_audit_scorer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Human audit scorer changed after the V6 freeze")
    if freeze.get("human_audit_config", {}).get("sha256") != sha256_file(args.config):
        raise RuntimeError("Human audit config changed after the V6 freeze")
    if audit.get("status") != "complete" or audit.get("rows") != int(
        config["samples_per_variant"]
    ) * len(config["formal_variants"]):
        raise RuntimeError("Human audit pack is incomplete")
    if audit.get("builder_sha256") != freeze.get("human_audit_builder_sha256"):
        raise RuntimeError("Human audit pack used a non-frozen builder")
    if audit.get("config", {}).get("sha256") != sha256_file(args.config):
        raise RuntimeError("Human audit pack used a different config")
    if not all(
        audit.get(key) is True
        for key in (
            "system_identity_blinded",
            "gold_labels_blinded",
            "automatic_scores_blinded",
        )
    ):
        raise RuntimeError("Human audit pack did not preserve required blinding")
    if audit.get("blind_csv", {}).get("sha256") != sha256_file(args.blind_csv):
        raise RuntimeError("Frozen blind worksheet changed")
    if audit.get("hidden_key", {}).get("sha256") != sha256_file(args.hidden_key):
        raise RuntimeError("Frozen hidden key changed")
    if automatic.get("status") != "complete" or automatic.get(
        "full_positive_claim_authorized"
    ) is not False:
        raise RuntimeError("Automatic V6 summary is missing or improperly authorizes a claim")
    if automatic.get("provenance", {}).get("freeze_manifest", {}).get(
        "sha256"
    ) != sha256_file(args.freeze_manifest):
        raise RuntimeError("Automatic summary and human audit use different freezes")

    columns, base_rows = read_csv(args.blind_csv)
    if tuple(columns) != BLIND_COLUMNS:
        raise ValueError("Blind worksheet columns differ from the frozen schema")
    base = indexed(base_rows, label="blind worksheet")
    first, first_id = validate_completed_sheet(
        args.annotator_one, base, label="annotator one"
    )
    second, second_id = validate_completed_sheet(
        args.annotator_two, base, label="annotator two"
    )
    adjudicated, adjudicator_id = validate_completed_sheet(
        args.adjudicated, base, label="adjudication"
    )
    if len({first_id, second_id, adjudicator_id}) != 3:
        raise ValueError("Annotator and adjudicator IDs must be distinct")
    final, disagreements = resolve_judgments(first, second, adjudicated)
    agreement = {
        field: {
            "raw_agreement": sum(
                first[audit_id][field] == second[audit_id][field] for audit_id in base
            )
            / len(base),
            "cohen_kappa": cohen_kappa(
                [first[audit_id][field] for audit_id in sorted(base)],
                [second[audit_id][field] for audit_id in sorted(base)],
            ),
            "disagreements": disagreements[field],
        }
        for field in JUDGMENT_FIELDS
    }

    hidden_rows = read_jsonl(args.hidden_key)
    hidden = {row["audit_id"]: row for row in hidden_rows}
    if len(hidden) != len(hidden_rows):
        raise ValueError("Hidden key contains duplicate audit IDs")
    if set(hidden) != set(base):
        raise ValueError("Hidden key and blind worksheet audit IDs differ")
    joined = [
        {
            **hidden[audit_id],
            "judgment": final[audit_id],
        }
        for audit_id in sorted(base)
    ]
    by_variant = {}
    positives_by_variant = {}
    for variant in config["formal_variants"]:
        rows = [row for row in joined if row["variant"] == variant]
        positives = [row for row in rows if row["prediction"] in POSITIVE]
        if len(rows) != int(config["samples_per_variant"]) or len(positives) != int(
            config["positive_outputs_per_variant"]
        ):
            raise ValueError(f"Human audit strata differ for {variant}")
        positives_by_variant[variant] = positives
        by_variant[variant] = {
            "audited_rows": len(rows),
            "positive_rows": len(positives),
            **mean_metrics(positives),
        }

    candidate = automatic["primary_method"]
    baseline = automatic["primary_baseline"]
    comparison = bootstrap_difference(
        positives_by_variant[candidate],
        positives_by_variant[baseline],
        replicates=int(config["human_comparison_bootstrap_replicates"]),
        confidence=float(config["confidence"]),
        seed=int(config["seed"]) + 1,
    )
    human_gates = {
        "claim_support": comparison["claim_support_rate"]["ci_low"] > 0.0,
        "rationale_faithfulness": comparison["rationale_faithfulness"]["ci_low"] > 0.0,
        "hallucination": comparison["hallucination_rate"]["ci_high"] < 0.0,
    }
    automatic_gates = automatic["primary_automatic_gates"]
    required_automatic = {
        key: bool(automatic_gates[key])
        for key in (
            "decision_quality",
            "unsafe_answer_noninferiority",
            "abstention_f1_noninferiority",
            "all_label_recalls_nonzero",
        )
    }
    authorized = all(required_automatic.values()) and all(human_gates.values())
    payload = {
        "status": "complete",
        "contract": "vgmtd_v6_blinded_human_grounding_summary",
        "primary_method": candidate,
        "primary_baseline": baseline,
        "role_is_secondary_ablation": True,
        "unclear_policy": config["unclear_policy"],
        "agreement": agreement,
        "adjudication": {
            "annotator_ids": [first_id, second_id],
            "adjudicator_id": adjudicator_id,
            "system_identity_unblinded_before_adjudication": False,
        },
        "metrics_by_variant": by_variant,
        "primary_human_comparison": comparison,
        "primary_human_gates": human_gates,
        "required_automatic_gates": required_automatic,
        "automatic_grounding_diagnostics_not_used_for_authorization": True,
        "full_positive_claim_authorized": authorized,
        "result_state": "positive-verifier-claim-authorized" if authorized else "positive-verifier-claim-not-authorized",
        "provenance": {
            "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
            "automatic_results_sha256": sha256_file(args.automatic_results),
            "audit_manifest_sha256": sha256_file(args.audit_manifest),
            "blind_csv_sha256": sha256_file(args.blind_csv),
            "annotator_one_sha256": sha256_file(args.annotator_one),
            "annotator_two_sha256": sha256_file(args.annotator_two),
            "adjudicated_sha256": sha256_file(args.adjudicated),
            "hidden_key_sha256": sha256_file(args.hidden_key),
            "config_sha256": sha256_file(args.config),
            "scorer_sha256": sha256_file(Path(__file__)),
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
