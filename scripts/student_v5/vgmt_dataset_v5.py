"""MiniMind dataset adapter for V5 hierarchical selective targets."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.student_v4_3.vgmt_dataset_v4_3 import (
    VGMTStructuredDecisionDatasetV43,
    decision_label_span,
)
from scripts.student_v5.inference_io_v5 import cap_conversations


CLAIM_TASK_ID = 0
QUERY_TASK_ID = 1
TARGET_CONTRACT_ID = "hierarchical_selective_targets_v5"
COMPATIBLE_TARGET_CONTRACT_IDS = frozenset(
    {TARGET_CONTRACT_ID, "verifier_core_targets_v6"}
)


class VGMTHierarchicalDatasetV5(VGMTStructuredDecisionDatasetV43):
    """Reuse V4.3 token alignment while replacing its flat target contract."""

    def __init__(self, jsonl_path, tokenizer, max_length=1024, max_prompt_tokens=640):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_prompt_tokens = int(max_prompt_tokens)
        if self.max_prompt_tokens <= 0 or self.max_prompt_tokens >= self.max_length:
            raise ValueError("max_prompt_tokens must lie strictly between zero and max_length")
        self.samples = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                conversations = row.get("conversations")
                if not isinstance(conversations, list) or len(conversations) != 3:
                    raise ValueError(f"Invalid conversations at line {line_number}")
                if [message.get("role") for message in conversations] != ["system", "user", "assistant"]:
                    raise ValueError(f"Invalid roles at line {line_number}")
                conversations, prompt_cap = cap_conversations(
                    tokenizer, conversations, self.max_prompt_tokens
                )
                metadata = row.get("metadata", {})
                source_id = metadata.get("source_example_id", f"line:{line_number}")
                task_type = metadata.get("task_type")
                if task_type not in {"claim_verification", "query_decision"}:
                    raise ValueError(f"Unknown task type for {source_id}: {task_type}")
                hard_label, _, _ = decision_label_span(conversations[2].get("content", ""), task_type)
                if metadata.get("contract_id") not in COMPATIBLE_TARGET_CONTRACT_IDS:
                    raise ValueError(f"Missing compatible hierarchical target contract for {source_id}")
                if metadata.get("decision_materialized_label") != hard_label:
                    raise ValueError(f"Materialized Decision mismatch for {source_id}")

                if task_type == "claim_verification":
                    sufficiency = float(metadata["p_sufficiency"])
                    polarity = float(metadata["p_support_given_sufficient"])
                    polarity_weight = float(metadata["polarity_loss_weight"])
                    answerability = 0.5
                    if metadata.get("p_answerable") is not None:
                        raise ValueError(f"Claim row has answerability target for {source_id}")
                else:
                    sufficiency = 0.5
                    polarity = 0.5
                    polarity_weight = 0.0
                    answerability = float(metadata["p_answerable"])
                    if metadata.get("p_sufficiency") is not None or metadata.get("p_support_given_sufficient") is not None:
                        raise ValueError(f"Query row has claim targets for {source_id}")
                values = (sufficiency, polarity, polarity_weight, answerability)
                if any(not torch.isfinite(torch.tensor(value)) or not 0.0 <= value <= 1.0 for value in values):
                    raise ValueError(f"Invalid hierarchical target for {source_id}")
                self.samples.append(
                    {
                        "conversations": conversations,
                        "source_id": source_id,
                        "task_type": task_type,
                        "task_id": CLAIM_TASK_ID if task_type == "claim_verification" else QUERY_TASK_ID,
                        "soft_target": [0.0, 0.0, 0.0],
                        "hierarchical_target": values,
                        "prompt_cap": prompt_cap,
                    }
                )
        if not self.samples:
            raise ValueError("V5 dataset is empty")
        self.decision_word_ids = list(tokenizer("Decision", add_special_tokens=False).input_ids)
        if not self.decision_word_ids:
            raise ValueError("Tokenizer gives an empty encoding for Decision")

    def __getitem__(self, index):
        input_ids, labels, decision_mask, task_id, _, attention_mask = super().__getitem__(index)
        sufficiency, polarity, polarity_weight, answerability = self.samples[index]["hierarchical_target"]
        return (
            input_ids,
            labels,
            decision_mask,
            task_id,
            torch.tensor(sufficiency, dtype=torch.float32),
            torch.tensor(polarity, dtype=torch.float32),
            torch.tensor(polarity_weight, dtype=torch.float32),
            torch.tensor(answerability, dtype=torch.float32),
            attention_mask,
        )


__all__ = [
    "COMPATIBLE_TARGET_CONTRACT_IDS",
    "TARGET_CONTRACT_ID",
    "VGMTHierarchicalDatasetV5",
]
