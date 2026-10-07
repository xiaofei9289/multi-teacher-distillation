#!/usr/bin/env python3
"""Verify the portable V6 training bundle, frozen MiniMind parent, and CUDA host."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import sha256_file  # noqa: E402


FORBIDDEN = (
    "human_gold_test",
    "sealed_test",
    "test_v2",
    "human_gold_v2.jsonl",
    "human_gold_dev",
)


def verify(bundle_root: Path, minimind_root: Path) -> dict:
    manifest_path = bundle_root / "SOURCE_MANIFEST_V6.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("package_id") != "vgmt_student_v6_verifier_core":
        raise RuntimeError("Unknown V6 handoff package")
    for item in manifest["files"]:
        path = bundle_root / item["path"]
        if not path.exists() or path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Missing or size-mismatched source: {item['path']}")
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Source hash mismatch: {item['path']}")
    forbidden = [
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
        and any(fragment in path.relative_to(bundle_root).as_posix().casefold() for fragment in FORBIDDEN)
    ]
    if forbidden:
        raise RuntimeError(f"Pre-test bundle contains forbidden evaluation data: {forbidden}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("V6 formal training requires exactly one visible CUDA GPU")
    recipe = json.loads(
        (bundle_root / "configs/student_training_v6.json").read_text(encoding="utf-8")
    )
    dtype = recipe.get("optimization", {}).get("dtype")
    if dtype not in {"float16", "bfloat16"}:
        raise RuntimeError(f"Unsupported formal training dtype: {dtype}")
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen bfloat16 recipe requires a CUDA device with bfloat16 support")
    base_weight = minimind_root / recipe["student"]["base_weight"]
    tokenizer = minimind_root / recipe["student"]["tokenizer"]
    model_code = minimind_root / "minimind/model/model_minimind.py"
    if not model_code.exists() or not tokenizer.exists() or not base_weight.exists():
        raise FileNotFoundError("Frozen MiniMind parent is incomplete")
    if sha256_file(base_weight) != recipe["student"]["base_weight_sha256"]:
        raise RuntimeError("Frozen MiniMind base checkpoint hash mismatch")
    free_bytes = shutil.disk_usage(bundle_root).free
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("At least 8 GiB free disk is required for 24 checkpoints")
    return {
        "status": "passed",
        "package_id": manifest["package_id"],
        "manifest_files": len(manifest["files"]),
        "test_labels_present": False,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "dtype": dtype,
        "bfloat16_supported": torch.cuda.is_bf16_supported(),
        "free_disk_bytes": free_bytes,
        "base_weight_sha256": sha256_file(base_weight),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=ROOT)
    parser.add_argument("--minimind-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle_root.resolve(), args.minimind_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
