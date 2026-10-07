#!/usr/bin/env python3
"""Freeze the complete V6 matrix and generated-grounding evaluation before test access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import sha256_file  # noqa: E402
from scripts.student_v5.preflight_supervision_v5 import validate_preflight_report  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "manifests/vgmt_v6_sealed_freeze.json"
    )
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/student_training_v6.json"
    )
    parser.add_argument(
        "--method-config", type=Path, default=ROOT / "configs/vgmt_v6.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    method = json.loads(args.method_config.read_text(encoding="utf-8"))
    if recipe.get("status") != "prospective_frozen_before_v6_training":
        raise RuntimeError("V6 recipe is not in its prospective frozen state")
    supervision_manifest_path = resolve(method["data"]["supervision_manifest"])
    supervision_manifest = json.loads(
        supervision_manifest_path.read_text(encoding="utf-8")
    )
    preflight_path = resolve(recipe["data"]["supervision_preflight"])
    preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    validate_preflight_report(
        preflight_path,
        recipe_path=args.recipe,
        supervision_manifest_path=supervision_manifest_path,
        tokenizer_path=Path(preflight_payload["tokenizer_path"]),
    )
    implementations = {
        "trainer_sha256": sha256_file(
            ROOT / "scripts/student_v5/train_hierarchical_student_v5.py"
        ),
        "dataset_sha256": sha256_file(ROOT / "scripts/student_v5/vgmt_dataset_v5.py"),
        "head_sha256": sha256_file(ROOT / "scripts/student_v5/hierarchical_head_v5.py"),
        "loss_sha256": sha256_file(ROOT / "scripts/student_v5/loss_contract_v5.py"),
        "inference_io_sha256": sha256_file(
            ROOT / "scripts/student_v5/inference_io_v5.py"
        ),
    }
    expected = {
        (variant, int(seed))
        for variant in recipe["variants"]
        for seed in recipe["optimization"]["seeds"]
    }
    if len(expected) != 24:
        raise RuntimeError("V6 freeze requires the exact 24-run matrix")
    found = {}
    for summary_path in args.runs_root.glob("**/training_summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        key = (summary.get("variant"), int(summary.get("seed", -1)))
        if key not in expected:
            raise ValueError(f"Unexpected formal run identity: {key}")
        if key in found:
            raise ValueError(f"Duplicate formal run identity: {key}")
        if summary.get("status") != "completed":
            raise ValueError(f"Incomplete formal run: {summary_path}")
        if summary.get("test_labels_loaded") or summary.get("legacy_dev_loaded"):
            raise ValueError(f"Forbidden evaluation data recorded by {summary_path}")
        if summary.get("recipe_id") != recipe["config_id"]:
            raise ValueError(f"Run used a different V6 recipe: {summary_path}")
        if summary.get("dtype") != recipe["optimization"]["dtype"]:
            raise ValueError(f"Run used a different numerical precision: {summary_path}")
        rows = int(recipe["data"]["rows_per_variant"])
        batch_size = int(recipe["optimization"]["batch_size"])
        accumulation = int(recipe["optimization"]["accumulation_steps"])
        micro_batches = (rows + batch_size - 1) // batch_size
        updates_per_epoch = (micro_batches + accumulation - 1) // accumulation
        expected_updates = int(recipe["optimization"]["epochs"]) * updates_per_epoch
        if int(summary.get("optimizer_updates", -1)) != expected_updates:
            raise ValueError(f"Run update budget mismatch: {summary_path}")
        if summary.get("files", {}).get("recipe_sha256") != sha256_file(args.recipe):
            raise ValueError(f"Recipe hash mismatch: {summary_path}")
        if any(
            summary.get("files", {}).get(name) != digest
            for name, digest in implementations.items()
        ):
            raise ValueError(f"Training implementation hash mismatch: {summary_path}")
        if summary.get("supervision_preflight", {}).get("sha256") != sha256_file(
            preflight_path
        ):
            raise ValueError(f"Run used a different V6 supervision preflight: {summary_path}")
        if summary.get("base_weight_sha256") != recipe["student"]["base_weight_sha256"]:
            raise ValueError(f"Base checkpoint mismatch: {summary_path}")
        if int(summary.get("data_rows", -1)) != int(recipe["data"]["rows_per_variant"]):
            raise ValueError(f"Training row budget mismatch: {summary_path}")
        data_path = Path(summary["data_path"])
        outputs = supervision_manifest.get("outputs", {})
        try:
            relative_data_path = data_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            relative_data_path = ""
        expected_data_hash = outputs.get(str(data_path)) or outputs.get(relative_data_path)
        if (
            not data_path.exists()
            or expected_data_hash != summary.get("data_sha256")
            or sha256_file(data_path) != expected_data_hash
        ):
            raise ValueError(f"Unauthorized or changed supervision: {summary_path}")
        checkpoint = Path(summary["output_weight"])
        if not checkpoint.exists() or sha256_file(checkpoint) != summary["output_weight_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        found[key] = {
            "summary": summary_path,
            "checkpoint": checkpoint,
            "checkpoint_sha256": summary["output_weight_sha256"],
        }
    if set(found) != expected:
        raise ValueError(f"V6 run matrix incomplete; missing={sorted(expected - set(found))}")

    required_pretest = {
        "gate_model": resolve(method["data"]["reused_gate_model"]),
        "gate_oof": resolve(method["data"]["reused_gate_oof"]),
        "role_report": resolve(method["data"]["reused_role_report"]),
        "supervision_manifest": supervision_manifest_path,
        "supervision_report": resolve(method["data"]["supervision_report"]),
        "supervision_preflight": resolve(recipe["data"]["supervision_preflight"]),
        "experiment_matrix": ROOT / "configs/experiment_matrix.yaml",
        "scope_contract": ROOT / "docs/planning/RESEARCH_SCOPE_FREEZE_V6.md",
        "method_contract": ROOT / "docs/method/VGMTD_METHOD_CONTRACT_V6.md",
        "student_contract": ROOT / "docs/method/VGMTD_STUDENT_CONTRACT_V6.md",
        "evaluation_contract": ROOT / "docs/method/VGMTD_EVALUATION_CONTRACT_V6.md",
    }
    for path in required_pretest.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing V6 pre-test artifact: {path}")
    role_report = json.loads(required_pretest["role_report"].read_text(encoding="utf-8"))
    if role_report.get("test_labels_loaded") or role_report.get("legacy_dev_loaded"):
        raise RuntimeError("Reused role report is ineligible for V6 freeze")

    predictor = ROOT / "scripts/student_v5/predict_hierarchical_student_v5.py"
    grounding_scorer = ROOT / "scripts/student_v6/score_generated_grounding_v6.py"
    grounding_evaluator = ROOT / "scripts/student_v6/evaluate_grounded_predictions_v6.py"
    sealed_prediction_launcher = (
        ROOT / "scripts/student_v6/run_sealed_prediction_matrix_v6.py"
    )
    matrix_summarizer = ROOT / "scripts/student_v6/summarize_grounded_matrix_v6.py"
    human_audit_builder = ROOT / "scripts/student_v6/build_blinded_grounding_audit_v6.py"
    human_audit_scorer = ROOT / "scripts/student_v6/score_blinded_grounding_audit_v6.py"
    human_audit_config = ROOT / "configs/human_grounding_audit_v6.json"
    payload = {
        "status": "frozen",
        "contract": "vgmtd_v6_one_shot_generated_grounding_evaluation",
        "test_open_authorized": True,
        "test_labels_loaded_during_freeze": False,
        "primary_method": method["evaluation"]["primary_method"],
        "primary_baseline": method["evaluation"]["primary_baseline"],
        "role_is_secondary_ablation": True,
        "recipe": {"path": str(args.recipe.resolve()), "sha256": sha256_file(args.recipe)},
        "method_config": {
            "path": str(args.method_config.resolve()),
            "sha256": sha256_file(args.method_config),
        },
        "pretest_artifacts": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in required_pretest.items()
        },
        "predictor_sha256": sha256_file(predictor),
        "grounding_scorer_sha256": sha256_file(grounding_scorer),
        "grounding_evaluator_sha256": sha256_file(grounding_evaluator),
        "sealed_prediction_launcher_sha256": sha256_file(sealed_prediction_launcher),
        "matrix_summarizer_sha256": sha256_file(matrix_summarizer),
        "human_audit_builder_sha256": sha256_file(human_audit_builder),
        "human_audit_scorer_sha256": sha256_file(human_audit_scorer),
        "human_audit_config": {
            "path": str(human_audit_config.resolve()),
            "sha256": sha256_file(human_audit_config),
        },
        "grounding_model": {
            "path": str(
                (ROOT / "data/results/models/domain_cross_encoder_nli_weak800").resolve()
            ),
            "model_sha256": sha256_file(
                ROOT
                / "data/results/models/domain_cross_encoder_nli_weak800/model.safetensors"
            ),
            "threshold": float(method["evaluation"]["support_probability_threshold"]),
        },
        "authorized_checkpoint_sha256": sorted(
            item["checkpoint_sha256"] for item in found.values()
        ),
        "runs": [
            {
                "variant": variant,
                "seed": seed,
                "summary": str(found[(variant, seed)]["summary"].resolve()),
                "summary_sha256": sha256_file(found[(variant, seed)]["summary"]),
                "checkpoint": str(found[(variant, seed)]["checkpoint"].resolve()),
                "checkpoint_sha256": found[(variant, seed)]["checkpoint_sha256"],
            }
            for variant, seed in sorted(found)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
