#!/usr/bin/env python3
"""Package V6 run records, predictions, audits, and reports without checkpoints or test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


FORBIDDEN = ("model.pth", "human_gold_test", "test_v2", "sealed_input")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    output = (args.output or root / "return_bundle/vgmt_student_v6_return.zip").resolve()
    summaries = sorted((root / "runs_v6").glob("*/*/training_summary.json"))
    if len(summaries) != 24:
        raise RuntimeError(f"Return package requires 24 training summaries, found {len(summaries)}")
    files = [root / "VERSION.json", root / "SOURCE_MANIFEST_V6.json"]
    files += summaries
    files += sorted((root / "runs_v6").glob("*/*/train_metrics.jsonl"))
    for relative in (
        "manifests/vgmt_v6_sealed_freeze.json",
        "reports/vgmt_v6_sealed_results.json",
        "reports/vgmt_v6_human_grounding_results.json",
        "reports/vgmt_v6_human_grounding_results_availability_amendment.json",
        "sealed_predictions/predictions_manifest.json",
        "human_audit/audit_manifest.json",
        "human_audit/blind_grounding_audit.csv",
        "human_audit/hidden_key.jsonl",
        "human_audit/annotator_1.csv",
        "human_audit/annotator_2.csv",
        "human_audit/adjudicated.csv",
        "configs/human_grounding_audit_v6_availability_amendment.json",
        "docs/planning/VGMTD_V6_HUMAN_AUDIT_AVAILABILITY_AMENDMENT.md",
        "docs/method/VGMTD_V6_HUMAN_GROUNDING_AUDIT_INSTRUCTIONS.md",
        "human_audit_availability_amendment/audit_manifest.json",
        "human_audit_availability_amendment/blind_grounding_audit.csv",
        "human_audit_availability_amendment/hidden_key.jsonl",
        "human_audit_availability_amendment/annotator_1.csv",
        "human_audit_availability_amendment/annotator_2.csv",
        "human_audit_availability_amendment/adjudicated.csv",
    ):
        path = root / relative
        if path.exists():
            files.append(path)
    for directory in (root / "sealed_predictions/raw", root / "sealed_predictions/grounded"):
        if directory.exists():
            files.extend(sorted(directory.rglob("*.jsonl")))
    for path in files:
        relative = path.relative_to(root).as_posix().casefold()
        if any(fragment in relative for fragment in FORBIDDEN):
            raise RuntimeError(f"Forbidden return payload: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root).as_posix())
    digest = sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(output), "sha256": digest, "files": len(set(files))}, indent=2))


if __name__ == "__main__":
    main()
