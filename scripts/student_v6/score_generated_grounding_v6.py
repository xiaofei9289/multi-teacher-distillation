#!/usr/bin/env python3
"""Score V6 generated answers and rationales against their supplied evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.score_vgmt_verifier_v3 import label_mapping, predict_probabilities  # noqa: E402
from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file, write_jsonl  # noqa: E402


POSITIVE_DECISIONS = frozenset({"Supported", "Refuted", "Answer"})
FIELD_PATTERN = re.compile(
    r"(?im)^\s*(Decision|Answer|Rationale|Citations)\s*:\s*"
)
SEALED_FRAGMENTS = ("human_gold_test", "test_v2", "sealed_test")


def parse_response(response: str) -> dict:
    text = str(response or "")
    matches = list(FIELD_PATTERN.finditer(text))
    fields: dict[str, str] = {}
    duplicate_fields = []
    for index, match in enumerate(matches):
        name = match.group(1).casefold()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        if name in fields:
            duplicate_fields.append(name)
        else:
            fields[name] = value
    raw_citations = fields.get("citations", "")
    citations = []
    if raw_citations and raw_citations.casefold() not in {"none", "n/a", "null"}:
        citations = [
            value.strip()
            for value in re.split(r"[,;\n]+", raw_citations)
            if value.strip()
        ]
    required = {"decision", "answer", "citations"}
    return {
        "decision": fields.get("decision", ""),
        "answer": fields.get("answer", ""),
        "rationale": fields.get("rationale", ""),
        "citations": citations,
        "duplicate_fields": sorted(set(duplicate_fields)),
        "parse_valid": required.issubset(fields) and not duplicate_fields,
    }


def _split_ids(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    return {
        item.strip()
        for item in re.split(r"[,;|\s]+", text)
        if item.strip()
    }


def allowed_evidence_ids(row: dict) -> set[str]:
    metadata = row.get("metadata", {})
    allowed = set()
    for value in (
        row.get("evidence_id"),
        row.get("evidence_ids"),
        metadata.get("evidence_id"),
        metadata.get("gold_evidence_ids"),
    ):
        allowed.update(_split_ids(value))
    return allowed


def citation_check(parsed: dict, allowed: set[str], positive: bool) -> dict:
    citations = set(parsed["citations"])
    invalid = sorted(citations - allowed)
    missing_required = bool(positive and allowed and not citations)
    valid = not invalid and not missing_required and (bool(allowed) or not citations)
    precision = (
        len(citations & allowed) / len(citations)
        if citations
        else (1.0 if not missing_required else 0.0)
    )
    return {
        "allowed_evidence_ids": sorted(allowed),
        "invalid_citations": invalid,
        "missing_required_citation": missing_required,
        "citation_valid": valid,
        "citation_precision": precision,
    }


def authorize_sealed_access(eval_path: Path, freeze_manifest: Path | None, sealed_test: bool) -> dict | None:
    looks_sealed = any(fragment in eval_path.as_posix().casefold() for fragment in SEALED_FRAGMENTS)
    if not looks_sealed and not sealed_test:
        return None
    if not sealed_test or freeze_manifest is None:
        raise RuntimeError("Sealed grounding scoring requires --sealed-test and --freeze-manifest")
    payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen" or payload.get("test_open_authorized") is not True:
        raise RuntimeError("Freeze manifest does not authorize sealed grounding scoring")
    if payload.get("grounding_scorer_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Grounding scorer changed after the sealed freeze")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "data/results/models/domain_cross_encoder_nli_weak800",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze = authorize_sealed_access(args.eval_data, args.freeze_manifest, args.sealed_test)
    model_path = args.model.resolve()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("Support threshold must lie in [0, 1]")
    predictions = read_jsonl(args.predictions)
    eval_rows = read_jsonl(args.eval_data)
    eval_by_id = {
        str(row.get("metadata", {}).get("source_example_id") or row.get("example_id")): row
        for row in eval_rows
    }
    if len(eval_by_id) != len(eval_rows):
        raise ValueError("Evaluation rows contain duplicate or empty source IDs")
    prediction_ids = [str(row.get("source_example_id", "")) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)) or set(prediction_ids) != set(eval_by_id):
        raise ValueError("Prediction and evaluation source IDs do not align one-to-one")

    prepared = []
    nli_pairs: list[tuple[str, str]] = []
    nli_targets: list[tuple[int, str]] = []
    for prediction in predictions:
        source_id = str(prediction["source_example_id"])
        source = eval_by_id[source_id]
        parsed = parse_response(prediction.get("response", ""))
        positive = prediction.get("prediction") in POSITIVE_DECISIONS
        citations = citation_check(parsed, allowed_evidence_ids(source), positive)
        item = {
            **prediction,
            "parsed_response": parsed,
            **citations,
            "automatic_grounding_scope": "positive_outputs_only",
            "answer_support_probability": None,
            "rationale_support_probability": None,
            "claim_support_pass": None,
            "rationale_faithful_pass": None,
            "automatic_hallucination": None,
        }
        prepared.append(item)
        if positive:
            evidence = str(source.get("evidence_text", ""))
            nli_pairs.append((evidence, parsed["answer"]))
            nli_targets.append((len(prepared) - 1, "answer"))
            nli_pairs.append((evidence, parsed["rationale"]))
            nli_targets.append((len(prepared) - 1, "rationale"))

    if nli_pairs:
        model = CrossEncoder(
            str(model_path),
            max_length=int(args.max_length),
            local_files_only=True,
        )
        mapping = label_mapping(model)
        probabilities = predict_probabilities(model, nli_pairs, int(args.batch_size))
        entail_index = mapping["entailment"]
        for (row_index, field), values in zip(nli_targets, probabilities):
            prepared[row_index][f"{field}_support_probability"] = float(values[entail_index])

    for item in prepared:
        if item["prediction"] not in POSITIVE_DECISIONS:
            continue
        answer_pass = (
            bool(item["parsed_response"]["answer"])
            and float(item["answer_support_probability"]) >= args.threshold
            and bool(item["citation_valid"])
        )
        rationale_pass = (
            bool(item["parsed_response"]["rationale"])
            and float(item["rationale_support_probability"]) >= args.threshold
            and item["parsed_response"]["decision"] == item["prediction"]
        )
        item["claim_support_pass"] = answer_pass
        item["rationale_faithful_pass"] = rationale_pass
        item["automatic_hallucination"] = not (
            answer_pass and item["parsed_response"]["parse_valid"]
        )
        item["support_probability_threshold"] = args.threshold
        item["automatic_metric_warning"] = (
            "Cross-encoder support is an automatic diagnostic and does not replace blinded human audit."
        )
        item["grounding_model_sha256"] = sha256_file(model_path / "model.safetensors")
        item["sealed_test"] = bool(freeze)

    write_jsonl(args.output, prepared)
    summary = {
        "status": "complete",
        "rows": len(prepared),
        "positive_rows": sum(row["prediction"] in POSITIVE_DECISIONS for row in prepared),
        "parse_valid_rows": sum(row["parsed_response"]["parse_valid"] for row in prepared),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "model": str(model_path),
        "model_sha256": sha256_file(model_path / "model.safetensors"),
        "threshold": args.threshold,
        "sealed_test": bool(freeze),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
