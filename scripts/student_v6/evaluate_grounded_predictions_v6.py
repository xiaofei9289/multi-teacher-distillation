#!/usr/bin/env python3
"""Evaluate decisions, grounding, hallucination, citations, and abstention for V6."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.evaluate_selective_v5 import evaluate as evaluate_decisions  # noqa: E402
from scripts.student_v5.vgmt_v5_contract import read_jsonl  # noqa: E402
from scripts.student_v6.score_generated_grounding_v6 import POSITIVE_DECISIONS  # noqa: E402


SAFE_LABELS = frozenset({"NotEnoughInfo", "Abstain"})


def _mean(values: list[float | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def abstention_metrics(rows: list[dict]) -> dict:
    tp = sum(row["gold_label"] in SAFE_LABELS and row["prediction"] in SAFE_LABELS for row in rows)
    fp = sum(row["gold_label"] not in SAFE_LABELS and row["prediction"] in SAFE_LABELS for row in rows)
    fn = sum(row["gold_label"] in SAFE_LABELS and row["prediction"] not in SAFE_LABELS for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "safe_action_tp": tp,
        "safe_action_fp": fp,
        "safe_action_fn": fn,
    }


def evaluate(rows: list[dict]) -> dict:
    decisions = evaluate_decisions(rows)
    positive = [row for row in rows if row["prediction"] in POSITIVE_DECISIONS]
    missing_grounding = [
        row["source_example_id"]
        for row in positive
        if row.get("claim_support_pass") is None
        or row.get("rationale_faithful_pass") is None
        or row.get("automatic_hallucination") is None
    ]
    if missing_grounding:
        raise ValueError(f"Positive rows lack frozen grounding scores: {missing_grounding[:5]}")
    pooled_unsafe = sum(
        (row["gold_label"] in SAFE_LABELS and row["prediction"] not in SAFE_LABELS)
        for row in rows
    )
    safe_gold = sum(row["gold_label"] in SAFE_LABELS for row in rows)
    automatic = {
        "positive_outputs": len(positive),
        "claim_support_rate": _mean([row["claim_support_pass"] for row in positive]),
        "rationale_faithfulness": _mean([row["rationale_faithful_pass"] for row in positive]),
        "hallucination_rate": _mean([row["automatic_hallucination"] for row in positive]),
        "evidence_citation_precision": _mean([row["citation_precision"] for row in positive]),
        "citation_valid_rate": _mean([row["citation_valid"] for row in positive]),
        "response_parse_rate": _mean([row["parsed_response"]["parse_valid"] for row in rows]),
        "mean_answer_support_probability": _mean(
            [row["answer_support_probability"] for row in positive]
        ),
        "mean_rationale_support_probability": _mean(
            [row["rationale_support_probability"] for row in positive]
        ),
        "warning": "Automatic cross-encoder grounding metrics require separate blinded human validation.",
    }
    return {
        **decisions,
        "answer_accuracy": decisions["accuracy"],
        "decision_macro_f1": decisions["macro_f1"],
        "unsafe_answer_rate": pooled_unsafe / safe_gold if safe_gold else 0.0,
        "unsafe_answer_numerator": pooled_unsafe,
        "unsafe_answer_denominator": safe_gold,
        **abstention_metrics(rows),
        "automatic_grounding": automatic,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(read_jsonl(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
