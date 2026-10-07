#!/usr/bin/env python3
"""Build the frozen, system-blinded V6 human grounding audit pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file, write_jsonl  # noqa: E402


POSITIVE = frozenset({"Supported", "Refuted", "Answer"})
SAFE = frozenset({"NotEnoughInfo", "Abstain"})
BLIND_COLUMNS = (
    "audit_id",
    "task_type",
    "query_text",
    "claim_text",
    "evidence_text",
    "allowed_evidence_ids",
    "model_decision",
    "model_answer",
    "model_rationale",
    "model_citations",
    "answer_supported",
    "rationale_faithful",
    "unsupported_material_claim",
    "citation_valid",
    "annotator_id",
    "notes",
)


def stable_key(seed: int, variant: str, stratum: str, source_id: str) -> str:
    value = f"{seed}|{variant}|{stratum}|{source_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def source_id(row: dict) -> str:
    value = row.get("metadata", {}).get("source_example_id") or row.get("example_id")
    if not value:
        raise ValueError("Evaluation row lacks source identity")
    return str(value)


def select(rows: list[dict], *, count: int, seed: int, variant: str, stratum: str) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Insufficient {stratum} rows for {variant}: {len(rows)} < {count}")
    return sorted(
        rows,
        key=lambda row: stable_key(seed, variant, stratum, str(row["source_example_id"])),
    )[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-manifest", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/human_grounding_audit_v6.json"
    )
    parser.add_argument("--blind-csv", type=Path, required=True)
    parser.add_argument("--hidden-key", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions_manifest.read_text(encoding="utf-8"))
    if freeze.get("human_audit_builder_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Human audit builder changed after the V6 freeze")
    frozen_config = freeze.get("human_audit_config", {})
    if frozen_config.get("sha256") != sha256_file(args.config):
        raise RuntimeError("Human audit config changed after the V6 freeze")
    if predictions.get("status") != "complete" or predictions.get("sealed_test") is not True:
        raise RuntimeError("Grounded prediction matrix is incomplete")
    if predictions.get("freeze_manifest", {}).get("sha256") != sha256_file(args.freeze_manifest):
        raise RuntimeError("Predictions were not produced under this freeze")
    eval_rows = read_jsonl(args.eval_data)
    eval_by_id = {source_id(row): row for row in eval_rows}
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("Evaluation rows contain duplicate source IDs")

    reference_seed = int(config["reference_training_seed"])
    records = {
        (row["variant"], int(row["seed"])): row for row in predictions["runs"]
    }
    chosen = []
    hidden = []
    seed = int(config["seed"])
    positive_count = int(config["positive_outputs_per_variant"])
    safe_count = int(config["safe_outputs_per_variant"])
    if positive_count + safe_count != int(config["samples_per_variant"]):
        raise ValueError("Human audit stratum counts do not match samples_per_variant")
    for variant in config["formal_variants"]:
        record = records.get((variant, reference_seed))
        if record is None:
            raise ValueError(f"Missing reference-seed grounded output for {variant}")
        grounded_path = Path(record["grounded_output"])
        if sha256_file(grounded_path) != record["grounded_output_sha256"]:
            raise RuntimeError(f"Grounded prediction changed: {grounded_path}")
        rows = read_jsonl(grounded_path)
        strata = {
            "positive": [row for row in rows if row["prediction"] in POSITIVE],
            "safe": [row for row in rows if row["prediction"] in SAFE],
        }
        selected = select(
            strata["positive"], count=positive_count, seed=seed, variant=variant, stratum="positive"
        ) + select(
            strata["safe"], count=safe_count, seed=seed, variant=variant, stratum="safe"
        )
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
                    "allowed_evidence_ids": ", ".join(row.get("allowed_evidence_ids", [])),
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
                    "gold_label": row["gold_label"],
                    "prediction": row["prediction"],
                    "automatic_claim_support_pass": row.get("claim_support_pass"),
                    "automatic_rationale_faithful_pass": row.get("rationale_faithful_pass"),
                    "automatic_hallucination": row.get("automatic_hallucination"),
                    "grounded_output_sha256": record["grounded_output_sha256"],
                }
            )
    chosen.sort(key=lambda row: stable_key(seed, "all", "shuffle", row["audit_id"]))
    if len({row["audit_id"] for row in chosen}) != len(chosen):
        raise ValueError("Audit IDs are not unique")
    args.blind_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.blind_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BLIND_COLUMNS)
        writer.writeheader()
        writer.writerows(chosen)
    write_jsonl(args.hidden_key, sorted(hidden, key=lambda row: row["audit_id"]))
    payload = {
        "status": "complete",
        "contract": config["config_id"],
        "rows": len(chosen),
        "variants": len(config["formal_variants"]),
        "reference_training_seed": reference_seed,
        "system_identity_blinded": True,
        "gold_labels_blinded": True,
        "automatic_scores_blinded": True,
        "blind_csv": {"path": str(args.blind_csv.resolve()), "sha256": sha256_file(args.blind_csv)},
        "hidden_key": {"path": str(args.hidden_key.resolve()), "sha256": sha256_file(args.hidden_key)},
        "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
        "builder_sha256": sha256_file(Path(__file__)),
        "test_labels_used_for_sampling": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
