"""Prompt rendering and deterministic prompt capping for V5 inference."""

from __future__ import annotations

import copy
import math
from typing import Any


SYSTEM_PROMPT = (
    "You are a small evidence-grounded recommendation model. Use only the "
    "provided evidence. If support is insufficient or conflicting, abstain."
)
EMPTY_THINK = "<think>\n\n</think>\n\n"
TRUNCATION_MARKER = "\n[... evidence truncated by the fixed prompt budget ...]\n"


def clean_empty_think(text: str) -> str:
    return text.replace(EMPTY_THINK, "")


def render_generation_prefix(tokenizer: Any, conversations: list[dict[str, Any]]) -> str:
    return tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True).input_ids)


def _prompt_token_count(tokenizer: Any, conversations: list[dict[str, Any]]) -> int:
    prefix = [dict(message) for message in conversations if message["role"] != "assistant"]
    return _token_count(tokenizer, render_generation_prefix(tokenizer, prefix))


def _shorten_evidence(tokenizer: Any, evidence: str, keep_tokens: int) -> str:
    evidence_ids = tokenizer(evidence, add_special_tokens=False).input_ids
    if keep_tokens >= len(evidence_ids):
        return evidence
    keep_tokens = max(0, keep_tokens)
    head_count = int(math.ceil(keep_tokens * 0.6))
    tail_count = keep_tokens - head_count
    head = tokenizer.decode(evidence_ids[:head_count], skip_special_tokens=True)
    tail = tokenizer.decode(evidence_ids[-tail_count:], skip_special_tokens=True) if tail_count else ""
    return head.rstrip() + TRUNCATION_MARKER + tail.lstrip()


def cap_conversations(
    tokenizer: Any,
    conversations: list[dict[str, Any]],
    max_prompt_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap evidence only, preserving the fixed system and task preamble."""
    capped = copy.deepcopy(conversations)
    before = _prompt_token_count(tokenizer, capped)
    if before <= max_prompt_tokens:
        return capped, {
            "truncated": False,
            "prompt_tokens_before": before,
            "prompt_tokens_after": before,
        }

    users = [index for index, message in enumerate(capped) if message["role"] == "user"]
    if len(users) != 1:
        raise ValueError("VG-MTD V5 prompt capping requires exactly one user message")
    user_index = users[0]
    content = capped[user_index]["content"]
    marker = "\nEvidence: "
    if marker not in content:
        raise ValueError("Long VG-MTD V5 prompt has no Evidence field")
    preamble, evidence = content.split(marker, 1)
    evidence_ids = tokenizer(evidence, add_special_tokens=False).input_ids

    low, high = 0, len(evidence_ids)
    best: list[dict[str, Any]] | None = None
    best_tokens = -1
    while low <= high:
        keep = (low + high) // 2
        candidate = copy.deepcopy(capped)
        candidate[user_index]["content"] = (
            preamble + marker + _shorten_evidence(tokenizer, evidence, keep)
        )
        count = _prompt_token_count(tokenizer, candidate)
        if count <= max_prompt_tokens:
            best = candidate
            best_tokens = count
            low = keep + 1
        else:
            high = keep - 1

    if best is None:
        raise ValueError(f"Prompt preamble alone exceeds the {max_prompt_tokens}-token cap")
    return best, {
        "truncated": True,
        "prompt_tokens_before": before,
        "prompt_tokens_after": best_tokens,
        "original_evidence_tokens": len(evidence_ids),
    }


def build_eval_conversations(row: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        f"Task type: {row['task_type']}\n"
        f"Domain: {row.get('domain', '')}\n"
        f"Query: {row.get('query_text', '')}\n"
        f"Claim: {row.get('claim_text', '')}\n"
        f"Evidence: {row.get('evidence_text', '')}\n"
        f"Evidence id: {row.get('evidence_id', row.get('metadata', {}).get('evidence_id', ''))}\n"
        f"Review id: {row.get('review_id', row.get('metadata', {}).get('review_id', ''))}\n"
        f"Time bin: {row.get('time_bin', row.get('metadata', {}).get('time_bin', ''))}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
