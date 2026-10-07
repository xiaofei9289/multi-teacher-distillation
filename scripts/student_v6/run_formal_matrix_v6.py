#!/usr/bin/env python3
"""Launch or audit the frozen 8-arm x 3-seed verifier-core V6 matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import sha256_file  # noqa: E402
from scripts.student_v5.preflight_supervision_v5 import (  # noqa: E402
    tokenizer_path_for,
    validate_preflight_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/student_training_v6.json"
    )
    parser.add_argument(
        "--supervision-manifest",
        type=Path,
        default=ROOT / "manifests/vgmt_supervision_v6_sha256.json",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=ROOT / "reports/vgmt_v6_supervision_preflight.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed_run(run_dir: Path, variant: str, seed: int, recipe_id: str) -> bool:
    summary_path = run_dir / "training_summary.json"
    weight_path = run_dir / "model.pth"
    if not summary_path.exists() and not weight_path.exists():
        return False
    if not summary_path.exists() or not weight_path.exists():
        raise RuntimeError(f"Partial formal run requires manual audit: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed":
        raise RuntimeError(f"Non-completed formal run requires manual audit: {run_dir}")
    if summary.get("recipe_id") != recipe_id:
        raise RuntimeError(f"Run used a different recipe: {run_dir}")
    if summary.get("variant") != variant or int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"Run identity mismatch: {run_dir}")
    if summary.get("test_labels_loaded") or summary.get("legacy_dev_loaded"):
        raise RuntimeError(f"Forbidden evaluation data recorded by run: {run_dir}")
    if sha256_file(weight_path) != summary.get("output_weight_sha256"):
        raise RuntimeError(f"Checkpoint hash mismatch: {run_dir}")
    return True


def validate_supervision(recipe: dict, manifest: dict) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    outputs = manifest.get("outputs", {})
    for variant in recipe["variants"]:
        path = (ROOT / recipe["data"]["supervision_root"] / variant / "train.jsonl").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing V6 supervision for {variant}: {path}")
        recorded = outputs.get(str(path)) or outputs.get(path.relative_to(ROOT).as_posix())
        if recorded is None or sha256_file(path) != recorded:
            raise RuntimeError(f"Supervision manifest does not authorize {path}")
        paths[variant] = path
    return paths


def main() -> None:
    args = parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if recipe.get("status") != "prospective_frozen_before_v6_training":
        raise RuntimeError("V6 recipe is not frozen")
    expected_count = len(recipe["variants"]) * len(recipe["optimization"]["seeds"])
    if expected_count != 24 or expected_count != int(recipe["optimization"]["formal_run_count"]):
        raise RuntimeError("V6 formal matrix must contain exactly 24 runs")
    manifest = json.loads(args.supervision_manifest.read_text(encoding="utf-8"))
    supervision = validate_supervision(recipe, manifest)
    validate_preflight_report(
        args.preflight_report.resolve(),
        recipe_path=args.recipe.resolve(),
        supervision_manifest_path=args.supervision_manifest.resolve(),
        tokenizer_path=tokenizer_path_for(args.minimind_root.resolve(), recipe),
    )
    plan = []
    for variant in recipe["variants"]:
        for seed in recipe["optimization"]["seeds"]:
            run_dir = (args.runs_root / variant / f"seed_{seed}").resolve()
            status = (
                "completed"
                if completed_run(run_dir, variant, int(seed), recipe["config_id"])
                else "pending"
            )
            plan.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "data_path": str(supervision[variant]),
                    "run_dir": str(run_dir),
                    "status": status,
                }
            )
    print(json.dumps({"formal_runs": expected_count, "plan": plan}, indent=2))
    if args.dry_run:
        return
    trainer = ROOT / "scripts/student_v5/train_hierarchical_student_v5.py"
    for item in plan:
        if item["status"] == "completed":
            continue
        subprocess.run(
            [
                sys.executable,
                str(trainer),
                "--variant",
                item["variant"],
                "--seed",
                str(item["seed"]),
                "--data-path",
                item["data_path"],
                "--run-dir",
                item["run_dir"],
                "--minimind-root",
                str(args.minimind_root.resolve()),
                "--recipe",
                str(args.recipe.resolve()),
                "--preflight-report",
                str(args.preflight_report.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
