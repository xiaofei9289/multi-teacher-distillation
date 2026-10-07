#!/usr/bin/env python3
"""Build the post-unsealing, availability-amended V6 blind audit pack."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file, write_jsonl  # noqa: E402
from scripts.student_v6.build_blinded_grounding_audit_v6 import (  # noqa: E402
    BLIND_COLUMNS,
    POSITIVE,
    SAFE,
    select,
    source_id,
    stable_key,
)


FROZEN_BUILDER = ROOT / "scripts/student_v6/build_blinded_grounding_audit_v6.py"
FROZEN_SCORER = ROOT / "scripts/student_v6/score_blinded_grounding_audit_v6.py"
FROZEN_CONFIG = ROOT / "configs/human_grounding_audit_v6.json"
DEFAULT_CONFIG = ROOT / "configs/human_grounding_audit_v6_availability_amendment.json"


def safe_selection(
    rows: list[dict], *, target: int, seed: int, variant: str
) -> list[dict]:
    """Select every available safe row up to the frozen target, without replacement."""
    count = min(target, len(rows))
    return sorted(
        rows,
        key=lambda row: stable_key(
            seed, variant, "safe", str(row["source_example_id"])
        ),
    )[:count]


def validate_amendment_contract(amended: dict, frozen: dict) -> None:
    if amended.get("parent_config_id") != frozen.get("config_id"):
        raise RuntimeError("Availability amendment names the wrong frozen parent")
    for key in (
        "seed",
        "reference_training_seed",
        "positive_outputs_per_variant",
        "formal_variants",
        "blind_fields",
        "judgment_fields",
        "allowed_judgments",
        "independent_annotators",
        "adjudicate_disagreements",
        "system_unblinding_before_adjudication",
        "human_comparison_bootstrap_replicates",
        "confidence",
        "unclear_policy",
    ):
        if amended.get(key) != frozen.get(key):
            raise RuntimeError(f"Availability amendment changed frozen field: {key}")
    if amended.get("safe_outputs_target_per_variant") != frozen.get(
        "safe_outputs_per_variant"
    ):
        raise RuntimeError("Availability amendment changed the frozen safe target")
    if amended.get("safe_sampling_policy") != "all_available_up_to_target_without_replacement":
        raise RuntimeError("Unexpected availability amendment sampling policy")
    if amended.get("confirmatory_positive_claim_authorization_allowed") is not False:
        raise RuntimeError("Availability amendment must prohibit confirmatory authorization")
    if amended.get("interpretation_scope") != "availability_amended_supportive_only":
        raise RuntimeError("Availability amendment has an unsafe interpretation scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-manifest", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--blind-csv", type=Path, required=True)
    parser.add_argument("--hidden-key", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen_config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions_manifest.read_text(encoding="utf-8"))
    validate_amendment_contract(config, frozen_config)

    if freeze.get("human_audit_builder_sha256") != sha256_file(FROZEN_BUILDER):
        raise RuntimeError("Frozen human audit builder changed after freeze")
    if freeze.get("human_audit_scorer_sha256") != sha256_file(FROZEN_SCORER):
        raise RuntimeError("Frozen human audit scorer changed after freeze")
    if freeze.get("human_audit_config", {}).get("sha256") != sha256_file(
        args.frozen_config
    ):
        raise RuntimeError("Frozen human audit config changed after freeze")
    if predictions.get("status") != "complete" or predictions.get("sealed_test") is not True:
        raise RuntimeError("Grounded prediction matrix is incomplete")
    if predictions.get("freeze_manifest", {}).get("sha256") != sha256_file(
        args.freeze_manifest
    ):
        raise RuntimeError("Predictions were not produced under this freeze")

    eval_rows = read_jsonl(args.eval_data)
    eval_by_id = {source_id(row): row for row in eval_rows}
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("Evaluation rows contain duplicate source IDs")

    reference_seed = int(config["reference_training_seed"])
    records = {
        (row["variant"], int(row["seed"])): row for row in predictions["runs"]
    }
    chosen: list[dict] = []
    hidden: list[dict] = []
    strata_by_variant: dict[str, dict[str, int]] = {}
    seed = int(config["seed"])
    positive_target = int(config["positive_outputs_per_variant"])
    safe_target = int(config["safe_outputs_target_per_variant"])

    for variant in config["formal_variants"]:
        record = records.get((variant, reference_seed))
        if record is None:
            raise ValueError(f"Missing reference-seed grounded output for {variant}")
        grounded_path = Path(record["grounded_output"])
        if sha256_file(grounded_path) != record["grounded_output_sha256"]:
            raise RuntimeError(f"Grounded prediction changed: {grounded_path}")
        rows = read_jsonl(grounded_path)
        positive_rows = [row for row in rows if row["prediction"] in POSITIVE]
        safe_rows = [row for row in rows if row["prediction"] in SAFE]
        positive_selected = select(
            positive_rows,
            count=positive_target,
            seed=seed,
            variant=variant,
            stratum="positive",
        )
        safe_selected = safe_selection(
            safe_rows, target=safe_target, seed=seed, variant=variant
        )
        strata_by_variant[variant] = {
            "positive_available": len(positive_rows),
            "positive_requested": positive_target,
            "positive_selected": len(positive_selected),
            "safe_available": len(safe_rows),
            "safe_requested": safe_target,
            "safe_selected": len(safe_selected),
            "safe_shortfall": safe_target - len(safe_selected),
        }
        for stratum, selected in (
            ("positive", positive_selected),
            ("safe", safe_selected),
        ):
            for row in selected:
                item_id = str(row["source_example_id"])
                source = eval_by_id[item_id]
                parsed = row["parsed_response"]
                audit_id = "audit_" + stable_key(seed, variant, "blind", item_id)[:20]
                chosen.append(
                    {
                        "audit_id": audit_id,
                        "task_type": row["task_type"],
                        "query_text": source.get("query_text", ""),
                        "claim_text": source.get("claim_text", ""),
                        "evidence_text": source.get("evidence_text", ""),
                        "allowed_evidence_ids": ", ".join(
                            row.get("allowed_evidence_ids", [])
                        ),
                        "model_decision": row["prediction"],
                        "model_answer": parsed.get("answer", ""),
                        "model_rationale": parsed.get("rationale", ""),
                        "model_citations": ", ".join(parsed.get("citations", [])),
                        "answer_supported": "",
                        "rationale_faithful": "",
                        "unsupported_material_claim": "",
                        "citation_valid": "",
                        "annotator_id": "",
                        "notes": "",
                    }
                )
                hidden.append(
                    {
                        "audit_id": audit_id,
                        "variant": variant,
                        "training_seed": reference_seed,
                        "source_example_id": item_id,
                        "sampling_stratum": stratum,
                        "gold_label": row["gold_label"],
                        "prediction": row["prediction"],
                        "automatic_claim_support_pass": row.get("claim_support_pass"),
                        "automatic_rationale_faithful_pass": row.get(
                            "rationale_faithful_pass"
                        ),
                        "automatic_hallucination": row.get("automatic_hallucination"),
                        "grounded_output_sha256": record["grounded_output_sha256"],
                    }
                )

    shortfalls = {
        variant: counts["safe_shortfall"]
        for variant, counts in strata_by_variant.items()
        if counts["safe_shortfall"] > 0
    }
    if not shortfalls:
        raise RuntimeError("Availability amendment is unnecessary: no safe-row shortfall")
    chosen.sort(key=lambda row: stable_key(seed, "all", "shuffle", row["audit_id"]))
    if len({row["audit_id"] for row in chosen}) != len(chosen):
        raise ValueError("Audit IDs are not unique")

    args.blind_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.blind_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(chosen)
    write_jsonl(args.hidden_key, sorted(hidden, key=lambda row: row["audit_id"]))

    requested_rows = (
        positive_target + safe_target
    ) * len(config["formal_variants"])
    payload = {
        "status": "complete",
        "contract": config["config_id"],
        "amendment_id": config["amendment_id"],
        "amendment_timing": config["amendment_timing"],
        "interpretation_scope": config["interpretation_scope"],
        "rows": len(chosen),
        "requested_rows_under_frozen_protocol": requested_rows,
        "variants": len(config["formal_variants"]),
        "reference_training_seed": reference_seed,
        "strata_by_variant": strata_by_variant,
        "safe_shortfalls": shortfalls,
        "positive_stratum_unchanged": True,
        "safe_control_stratum_only_is_availability_capped": True,
        "system_identity_blinded": True,
        "gold_labels_blinded": True,
        "automatic_scores_blinded": True,
        "confirmatory_positive_claim_authorization_prohibited": True,
        "blind_csv": {
            "path": str(args.blind_csv.resolve()),
            "sha256": sha256_file(args.blind_csv),
        },
        "hidden_key": {
            "path": str(args.hidden_key.resolve()),
            "sha256": sha256_file(args.hidden_key),
        },
        "amendment_config": {
            "path": str(args.config.resolve()),
            "sha256": sha256_file(args.config),
        },
        "frozen_parent": {
            "config_path": str(args.frozen_config.resolve()),
            "config_sha256": sha256_file(args.frozen_config),
            "builder_sha256": sha256_file(FROZEN_BUILDER),
            "scorer_sha256": sha256_file(FROZEN_SCORER),
        },
        "builder_sha256": sha256_file(Path(__file__)),
        "test_labels_used_for_sampling": False,
        "automatic_scores_used_for_sampling": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
