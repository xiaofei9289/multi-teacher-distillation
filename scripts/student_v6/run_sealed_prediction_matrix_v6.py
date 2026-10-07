#!/usr/bin/env python3
"""Generate and verifier-score the exact frozen 24-run V6 sealed matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/student_training_v6.json"
    )
    parser.add_argument(
        "--method-config", type=Path, default=ROOT / "configs/vgmt_v6.json"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expected_keys(recipe: dict) -> set[tuple[str, int]]:
    return {
        (variant, int(seed))
        for variant in recipe["variants"]
        for seed in recipe["optimization"]["seeds"]
    }


def valid_prediction(path: Path, checkpoint_sha256: str, examples: int) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    ids = [str(row.get("source_example_id", "")) for row in rows]
    return (
        len(rows) == examples
        and len(ids) == len(set(ids))
        and all(ids)
        and all(row.get("sealed_test") is True for row in rows)
        and all(row.get("weight_sha256") == checkpoint_sha256 for row in rows)
    )


def valid_grounding(path: Path, checkpoint_sha256: str, examples: int, model_sha256: str) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    ids = [str(row.get("source_example_id", "")) for row in rows]
    positive = {"Supported", "Refuted", "Answer"}
    return (
        len(rows) == examples
        and len(ids) == len(set(ids))
        and all(ids)
        and all(row.get("sealed_test") is True for row in rows)
        and all(row.get("weight_sha256") == checkpoint_sha256 for row in rows)
        and all(
            row.get("grounding_model_sha256") == model_sha256
            for row in rows
            if row.get("prediction") in positive
        )
    )


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    method = json.loads(args.method_config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or freeze.get("test_open_authorized") is not True:
        raise RuntimeError("The supplied V6 manifest does not authorize sealed prediction")
    if freeze.get("sealed_prediction_launcher_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("V6 sealed launcher changed after the freeze")
    if freeze.get("recipe", {}).get("sha256") != sha256_file(args.recipe):
        raise RuntimeError("V6 recipe differs from the frozen manifest")
    if freeze.get("method_config", {}).get("sha256") != sha256_file(args.method_config):
        raise RuntimeError("V6 method config differs from the frozen manifest")
    if freeze.get("primary_method") != method["evaluation"]["primary_method"]:
        raise RuntimeError("Frozen primary method differs from V6 config")
    if freeze.get("primary_baseline") != method["evaluation"]["primary_baseline"]:
        raise RuntimeError("Frozen primary baseline differs from V6 config")
    expected = expected_keys(recipe)
    if len(expected) != 24:
        raise RuntimeError("V6 sealed matrix must contain exactly 24 runs")
    frozen_runs = {
        (row["variant"], int(row["seed"])): row for row in freeze.get("runs", [])
    }
    if set(frozen_runs) != expected:
        raise RuntimeError("Freeze manifest does not contain the exact V6 matrix")

    examples = int(method["evaluation"]["sealed_test_examples"])
    model_sha256 = str(freeze["grounding_model"]["model_sha256"])
    output_manifest = args.output_manifest or args.predictions_root / "predictions_manifest.json"
    plan = []
    for variant in recipe["variants"]:
        for seed in recipe["optimization"]["seeds"]:
            frozen = frozen_runs[(variant, int(seed))]
            checkpoint = Path(frozen["checkpoint"])
            if not checkpoint.exists() or sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
                raise RuntimeError(f"Frozen checkpoint is missing or changed: {checkpoint}")
            raw_output = args.predictions_root / "raw" / variant / f"seed_{seed}.jsonl"
            grounded_output = args.predictions_root / "grounded" / variant / f"seed_{seed}.jsonl"
            raw_complete = valid_prediction(
                raw_output, frozen["checkpoint_sha256"], examples
            )
            grounded_complete = valid_grounding(
                grounded_output,
                frozen["checkpoint_sha256"],
                examples,
                model_sha256,
            )
            if grounded_complete and not raw_complete:
                raise RuntimeError(f"Grounded output exists without valid raw prediction: {grounded_output}")
            plan.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": frozen["checkpoint_sha256"],
                    "raw_output": str(raw_output),
                    "grounded_output": str(grounded_output),
                    "raw_status": "complete" if raw_complete else "pending",
                    "grounding_status": "complete" if grounded_complete else "pending",
                }
            )
    print(json.dumps({"sealed_examples": examples, "formal_predictions": len(plan), "plan": plan}, indent=2))
    if args.dry_run:
        return

    predictor = ROOT / "scripts/student_v5/predict_hierarchical_student_v5.py"
    grounding_scorer = ROOT / "scripts/student_v6/score_generated_grounding_v6.py"
    threshold = float(freeze["grounding_model"]["threshold"])
    model_path = Path(freeze["grounding_model"]["path"])
    if sha256_file(model_path / "model.safetensors") != model_sha256:
        raise RuntimeError("Frozen grounding model changed")
    for item in plan:
        raw_output = Path(item["raw_output"])
        grounded_output = Path(item["grounded_output"])
        if item["raw_status"] != "complete":
            raw_output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(predictor),
                "--weight",
                item["checkpoint"],
                "--eval-data",
                str(args.eval_data.resolve()),
                "--output",
                str(raw_output.resolve()),
                "--minimind-root",
                str(args.minimind_root.resolve()),
                "--recipe",
                str(args.recipe.resolve()),
                "--sealed-test",
                "--freeze-manifest",
                str(args.freeze_manifest.resolve()),
            ]
            if args.tokenizer_path:
                command.extend(["--tokenizer-path", str(args.tokenizer_path.resolve())])
            subprocess.run(command, cwd=ROOT, check=True)
        if not valid_prediction(raw_output, item["checkpoint_sha256"], examples):
            raise RuntimeError(f"Raw prediction failed validation: {raw_output}")
        if item["grounding_status"] != "complete":
            grounded_output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    str(grounding_scorer),
                    "--predictions",
                    str(raw_output.resolve()),
                    "--eval-data",
                    str(args.eval_data.resolve()),
                    "--output",
                    str(grounded_output.resolve()),
                    "--model",
                    str(model_path.resolve()),
                    "--threshold",
                    str(threshold),
                    "--sealed-test",
                    "--freeze-manifest",
                    str(args.freeze_manifest.resolve()),
                ],
                cwd=ROOT,
                check=True,
            )
        if not valid_grounding(
            grounded_output, item["checkpoint_sha256"], examples, model_sha256
        ):
            raise RuntimeError(f"Grounded prediction failed validation: {grounded_output}")

    manifest = {
        "status": "complete",
        "contract": "vgmtd_v6_exact_sealed_generated_grounding_matrix",
        "sealed_test": True,
        "test_open_count": 1,
        "examples_per_run": examples,
        "primary_method": method["evaluation"]["primary_method"],
        "primary_baseline": method["evaluation"]["primary_baseline"],
        "role_is_secondary_ablation": True,
        "freeze_manifest": {
            "path": str(args.freeze_manifest.resolve()),
            "sha256": sha256_file(args.freeze_manifest),
        },
        "eval_data": {
            "path": str(args.eval_data.resolve()),
            "sha256": sha256_file(args.eval_data),
        },
        "launcher_sha256": sha256_file(Path(__file__)),
        "predictor_sha256": sha256_file(predictor),
        "grounding_scorer_sha256": sha256_file(grounding_scorer),
        "grounding_model_sha256": model_sha256,
        "support_probability_threshold": threshold,
        "runs": [],
    }
    for item in plan:
        raw_output = Path(item["raw_output"])
        grounded_output = Path(item["grounded_output"])
        manifest["runs"].append(
            {
                **{
                    key: item[key]
                    for key in ("variant", "seed", "checkpoint", "checkpoint_sha256")
                },
                "raw_output": str(raw_output.resolve()),
                "raw_output_sha256": sha256_file(raw_output),
                "grounded_output": str(grounded_output.resolve()),
                "grounded_output_sha256": sha256_file(grounded_output),
            }
        )
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(output_manifest),
                "sha256": sha256_file(output_manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
