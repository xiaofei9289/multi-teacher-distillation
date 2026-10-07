#!/usr/bin/env python3
"""Exhaustively tokenize and authorize all eight frozen V5 supervision arms."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_dataset_v5 import VGMTHierarchicalDatasetV5  # noqa: E402
from scripts.student_v5.vgmt_v5_contract import sha256_file  # noqa: E402


DATASET_PATH = ROOT / "scripts/student_v5/vgmt_dataset_v5.py"
INFERENCE_IO_PATH = ROOT / "scripts/student_v5/inference_io_v5.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument(
        "--recipe", type=Path, default=ROOT / "configs/student_training_v5.json"
    )
    parser.add_argument(
        "--supervision-manifest",
        type=Path,
        default=ROOT / "manifests/vgmt_supervision_v5_sha256.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/vgmt_v5_supervision_preflight.json",
    )
    return parser.parse_args()


def supervision_paths(recipe: dict) -> dict[str, Path]:
    root = ROOT / recipe["data"]["supervision_root"]
    return {
        variant: (root / variant / "train.jsonl").resolve()
        for variant in recipe["variants"]
    }


def tokenizer_path_for(minimind_root: Path, recipe: dict) -> Path:
    return (minimind_root / recipe["student"]["tokenizer"]).resolve()


def manifest_digest_for(path: Path, manifest: dict) -> str | None:
    outputs = manifest.get("outputs", {})
    candidates = (str(path), str(path.resolve()))
    for candidate in candidates:
        if candidate in outputs:
            return str(outputs[candidate])
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = ""
    return str(outputs[relative]) if relative and relative in outputs else None


def validate_preflight_report(
    report_path: Path,
    *,
    recipe_path: Path,
    supervision_manifest_path: Path,
    tokenizer_path: Path,
) -> dict:
    """Fail closed unless the report matches every runtime input and implementation."""
    recipe_path = recipe_path.resolve()
    supervision_manifest_path = supervision_manifest_path.resolve()
    tokenizer_path = tokenizer_path.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    manifest = json.loads(supervision_manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(f"V5 supervision preflight did not pass: {report_path}")
    if report.get("recipe_id") != recipe["config_id"]:
        raise RuntimeError("V5 supervision preflight recipe id mismatch")
    if report.get("recipe_sha256") != sha256_file(recipe_path):
        raise RuntimeError("V5 supervision preflight recipe hash mismatch")
    if report.get("supervision_manifest_sha256") != sha256_file(supervision_manifest_path):
        raise RuntimeError("V5 supervision preflight manifest hash mismatch")
    if Path(report.get("tokenizer_path", "")).resolve() != tokenizer_path:
        raise RuntimeError("V5 supervision preflight tokenizer path mismatch")
    implementations = report.get("implementation_sha256", {})
    if implementations.get("dataset") != sha256_file(DATASET_PATH):
        raise RuntimeError("V5 supervision preflight dataset implementation changed")
    if implementations.get("inference_io") != sha256_file(INFERENCE_IO_PATH):
        raise RuntimeError("V5 supervision preflight prompt-cap implementation changed")
    optimization = recipe["optimization"]
    if int(report.get("max_seq_len", -1)) != int(optimization["max_seq_len"]):
        raise RuntimeError("V5 supervision preflight max_seq_len mismatch")
    if int(report.get("max_prompt_tokens", -1)) != int(optimization["max_prompt_tokens"]):
        raise RuntimeError("V5 supervision preflight prompt budget mismatch")
    if report.get("test_labels_loaded") or report.get("legacy_dev_loaded"):
        raise RuntimeError("V5 supervision preflight recorded forbidden evaluation access")

    paths = supervision_paths(recipe)
    variants = report.get("variants", {})
    if set(variants) != set(paths):
        raise RuntimeError("V5 supervision preflight variant set mismatch")
    expected_rows = int(recipe["data"]["rows_per_variant"])
    expected_sources = int(recipe["data"]["unique_source_examples"])
    for variant, path in paths.items():
        item = variants[variant]
        recorded = manifest_digest_for(path, manifest)
        actual = sha256_file(path)
        if recorded is None or recorded != actual or item.get("data_sha256") != actual:
            raise RuntimeError(f"V5 supervision preflight data hash mismatch for {variant}")
        if Path(item.get("data_path", "")).resolve() != path:
            raise RuntimeError(f"V5 supervision preflight data path mismatch for {variant}")
        if item.get("status") != "passed":
            raise RuntimeError(f"V5 supervision preflight arm did not pass: {variant}")
        if int(item.get("rows_scanned", -1)) != expected_rows:
            raise RuntimeError(f"V5 supervision preflight row count mismatch for {variant}")
        if int(item.get("unique_source_examples", -1)) != expected_sources:
            raise RuntimeError(f"V5 supervision preflight source count mismatch for {variant}")
        if int(item.get("max_prompt_tokens_after", -1)) > int(optimization["max_prompt_tokens"]):
            raise RuntimeError(f"V5 prompt budget exceeded after capping for {variant}")
        if int(item.get("max_sequence_tokens", -1)) > int(optimization["max_seq_len"]):
            raise RuntimeError(f"V5 sequence budget exceeded after capping for {variant}")
    expected_total = expected_rows * len(paths)
    if int(report.get("rows_scanned_total", -1)) != expected_total:
        raise RuntimeError("V5 supervision preflight total row count mismatch")
    return report


def scan_variant(
    variant: str,
    path: Path,
    tokenizer,
    *,
    expected_rows: int,
    expected_sources: int,
    max_seq_len: int,
    max_prompt_tokens: int,
) -> dict:
    dataset = VGMTHierarchicalDatasetV5(
        path,
        tokenizer,
        max_length=max_seq_len,
        max_prompt_tokens=max_prompt_tokens,
    )
    if len(dataset) != expected_rows:
        raise RuntimeError(f"{variant} has {len(dataset)} rows; expected {expected_rows}")
    source_ids = {sample["source_id"] for sample in dataset.samples}
    if len(source_ids) != expected_sources:
        raise RuntimeError(
            f"{variant} has {len(source_ids)} source examples; expected {expected_sources}"
        )

    max_sequence = 0
    min_sequence = max_seq_len
    max_prompt_before = 0
    max_prompt_after = 0
    truncated_rows = 0
    truncated_sources: set[str] = set()
    longest_source = ""
    for index in range(len(dataset)):
        item = dataset[index]
        sequence_tokens = int(item[-1].sum().item())
        sample = dataset.samples[index]
        cap = sample["prompt_cap"]
        if sequence_tokens > max_sequence:
            max_sequence = sequence_tokens
            longest_source = sample["source_id"]
        min_sequence = min(min_sequence, sequence_tokens)
        max_prompt_before = max(max_prompt_before, int(cap["prompt_tokens_before"]))
        max_prompt_after = max(max_prompt_after, int(cap["prompt_tokens_after"]))
        if cap["truncated"]:
            truncated_rows += 1
            truncated_sources.add(sample["source_id"])

    return {
        "status": "passed",
        "data_path": str(path),
        "data_sha256": sha256_file(path),
        "rows_scanned": len(dataset),
        "unique_source_examples": len(source_ids),
        "truncated_rows": truncated_rows,
        "truncated_source_examples": len(truncated_sources),
        "max_prompt_tokens_before": max_prompt_before,
        "max_prompt_tokens_after": max_prompt_after,
        "min_sequence_tokens": min_sequence,
        "max_sequence_tokens": max_sequence,
        "longest_sequence_source_example_id": longest_source,
    }


def main() -> None:
    args = parse_args()
    recipe_path = args.recipe.resolve()
    manifest_path = args.supervision_manifest.resolve()
    output_path = args.output.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("status") not in {
        "prospective_frozen_before_v5_training",
        "prospective_frozen_before_v6_training",
    }:
        raise RuntimeError("This scanner requires a compatible frozen student recipe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_path = tokenizer_path_for(args.minimind_root.resolve(), recipe)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    paths = supervision_paths(recipe)
    expected_rows = int(recipe["data"]["rows_per_variant"])
    expected_sources = int(recipe["data"]["unique_source_examples"])
    max_seq_len = int(recipe["optimization"]["max_seq_len"])
    max_prompt_tokens = int(recipe["optimization"]["max_prompt_tokens"])
    payload = {
        "status": "running",
        "contract": f"{recipe['config_id']}_exhaustive_supervision_preflight",
        "recipe_id": recipe["config_id"],
        "recipe_sha256": sha256_file(recipe_path),
        "supervision_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_path": str(tokenizer_path),
        "max_seq_len": max_seq_len,
        "max_prompt_tokens": max_prompt_tokens,
        "implementation_sha256": {
            "dataset": sha256_file(DATASET_PATH),
            "inference_io": sha256_file(INFERENCE_IO_PATH),
        },
        "variants": {},
        "rows_scanned_total": 0,
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
    }
    try:
        for variant, path in paths.items():
            recorded = manifest_digest_for(path, manifest)
            if recorded is None or sha256_file(path) != recorded:
                raise RuntimeError(f"Supervision manifest does not authorize {variant}")
            payload["variants"][variant] = scan_variant(
                variant,
                path,
                tokenizer,
                expected_rows=expected_rows,
                expected_sources=expected_sources,
                max_seq_len=max_seq_len,
                max_prompt_tokens=max_prompt_tokens,
            )
            payload["rows_scanned_total"] += expected_rows
        payload["status"] = "passed"
    except Exception as exc:
        payload["status"] = "failed"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_preflight_report(
        output_path,
        recipe_path=recipe_path,
        supervision_manifest_path=manifest_path,
        tokenizer_path=tokenizer_path,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
