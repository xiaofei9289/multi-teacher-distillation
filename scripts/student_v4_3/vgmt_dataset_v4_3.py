"""MiniMind SFT dataset for the V4.3 structured soft-decision objective."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from .decision_contract_v4_2 import decision_label_span
except ImportError:  # Project source tree; the desktop bundle copies the contract here.
    from scripts.student_v4_2.decision_contract_v4_2 import decision_label_span


CLAIM_LABELS = ("Supported", "Refuted", "NotEnoughInfo")
QUERY_LABELS = ("Answer", "Abstain")
ALL_LABELS = CLAIM_LABELS + QUERY_LABELS
CLAIM_TASK_ID = 0
QUERY_TASK_ID = 1
TARGET_CONTRACT_ID = "structured_soft_decision_v4_3"


class VGMTStructuredDecisionDatasetV43(Dataset):
    """Return prefix-aligned LM labels plus a leakage-safe soft Decision target."""

    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.samples = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                conversations = row.get("conversations")
                if not isinstance(conversations, list) or len(conversations) != 3:
                    raise ValueError(f"Invalid conversations at line {line_number}")
                if [message.get("role") for message in conversations] != [
                    "system",
                    "user",
                    "assistant",
                ]:
                    raise ValueError(f"Invalid roles at line {line_number}")
                metadata = row.get("metadata", {})
                source_id = metadata.get("source_example_id", f"line:{line_number}")
                task_type = metadata.get("task_type")
                if task_type is None:
                    first_user_line = conversations[1].get("content", "").splitlines()[0]
                    prefix = "Task type: "
                    if not first_user_line.startswith(prefix):
                        raise ValueError(f"Missing task type for {source_id}")
                    task_type = first_user_line[len(prefix):].strip()
                if task_type not in {"claim_verification", "query_decision"}:
                    raise ValueError(f"Unknown task type for {source_id}: {task_type}")
                hard_label, _, _ = decision_label_span(
                    conversations[2].get("content", ""), task_type
                )
                if metadata.get("decision_target_contract") != TARGET_CONTRACT_ID:
                    raise ValueError(f"Missing V4.3 soft-target contract for {source_id}")
                distribution = metadata.get("decision_target_distribution")
                if not isinstance(distribution, dict) or set(distribution) != set(ALL_LABELS):
                    raise ValueError(f"Invalid soft-target keys for {source_id}")
                values = {label: float(distribution[label]) for label in ALL_LABELS}
                if any(not torch.isfinite(torch.tensor(value)) or value < 0 for value in values.values()):
                    raise ValueError(f"Invalid soft-target value for {source_id}")
                legal = CLAIM_LABELS if task_type == "claim_verification" else QUERY_LABELS
                illegal = set(ALL_LABELS) - set(legal)
                if any(values[label] != 0.0 for label in illegal):
                    raise ValueError(f"Illegal label mass for {source_id}")
                if abs(sum(values.values()) - 1.0) > 1e-6:
                    raise ValueError(f"Soft target does not sum to one for {source_id}")
                if metadata.get("decision_materialized_label") != hard_label:
                    raise ValueError(f"Materialized Decision mismatch for {source_id}")
                padded_target = (
                    [values[label] for label in CLAIM_LABELS]
                    if task_type == "claim_verification"
                    else [values[label] for label in QUERY_LABELS] + [0.0]
                )
                self.samples.append(
                    {
                        "conversations": conversations,
                        "source_id": source_id,
                        "task_type": task_type,
                        "task_id": CLAIM_TASK_ID if task_type == "claim_verification" else QUERY_TASK_ID,
                        "soft_target": padded_target,
                    }
                )
        if not self.samples:
            raise ValueError("V4.3 dataset is empty")
        self.decision_word_ids = list(
            tokenizer("Decision", add_special_tokens=False).input_ids
        )
        if not self.decision_word_ids:
            raise ValueError("Tokenizer gives an empty encoding for Decision")

    def __len__(self):
        return len(self.samples)

    def _render(self, conversations):
        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _render_generation_prefix(self, conversations, source_id):
        prompt_messages = conversations[:2]
        try:
            prefix = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot render the frozen open_thinking=False prefix for {source_id}"
            ) from exc
        if not isinstance(prefix, str) or not prefix:
            raise RuntimeError(f"Empty generation prefix for {source_id}")
        return prefix

    def _aligned_prefix_length(
        self,
        rendered,
        generation_prefix,
        assistant_content,
        input_ids,
        offsets,
        source_id,
    ):
        """Return the exact inference-prefix token count or fail closed.

        MiniMind's chat template materializes an empty ``<think>`` scaffold
        when ``open_thinking=False``.  That scaffold belongs to the prompt at
        inference, so it must be masked together with system/user/template
        tokens during training.  Character-prefix agreement alone is not
        sufficient because a tokenizer could merge across the boundary; the
        token IDs must also be an exact prefix of the full training render.
        """

        if not rendered.startswith(generation_prefix):
            raise RuntimeError(
                f"Training render does not start with the frozen generation prefix for {source_id}"
            )
        if not rendered[len(generation_prefix):].startswith(assistant_content):
            raise RuntimeError(
                f"Assistant target does not immediately follow the generation prefix for {source_id}"
            )
        prefix_ids = list(
            self.tokenizer(generation_prefix, add_special_tokens=True).input_ids
        )
        if not prefix_ids or len(prefix_ids) >= len(input_ids):
            raise RuntimeError(f"Invalid tokenized generation prefix for {source_id}")
        if input_ids[:len(prefix_ids)] != prefix_ids:
            raise RuntimeError(
                f"Training and evaluation prefix token IDs differ for {source_id}"
            )
        first_target = len(prefix_ids)
        if input_ids[
            first_target:first_target + len(self.decision_word_ids)
        ] != self.decision_word_ids:
            raise RuntimeError(
                f"First supervised token sequence is not Decision for {source_id}"
            )
        start, end = offsets[first_target]
        if start != len(generation_prefix) or end <= start:
            raise RuntimeError(
                f"Tokenizer offset crosses the prompt/Decision boundary for {source_id}"
            )
        if not rendered[start:end] or rendered[start:end][0] != "D":
            raise RuntimeError(
                f"First supervised token does not begin at Decision for {source_id}"
            )
        return first_target

    @staticmethod
    def _generate_labels(input_ids, first_target):
        if not 0 < first_target < len(input_ids):
            raise ValueError("first_target must lie inside the rendered token sequence")
        # Exactly the inference prefix—including the fixed empty-think
        # scaffold—is ignored.  The target begins at the first Decision token
        # and continues through the assistant terminator.
        return [-100] * first_target + list(input_ids[first_target:])

    def _decision_mask(
        self,
        rendered: str,
        conversations: list[dict],
        offsets: list[tuple[int, int]],
        labels: list[int],
        task_type: str,
        source_id: str,
    ) -> list[int]:
        assistant_content = conversations[2]["content"]
        # Validate the label contract, then mask the complete deterministic
        # first line.  At inference the head supplies this whole line, so no
        # generative token on it may compete with the structured objective.
        decision_label_span(assistant_content, task_type)
        newline = assistant_content.find("\n")
        if newline < 0:
            raise ValueError(f"Decision line has no terminator for {source_id}")
        local_start, local_end = 0, newline + 1
        assistant_start = rendered.rfind(assistant_content)
        if assistant_start < 0 or rendered.find(assistant_content) != assistant_start:
            raise ValueError(f"Assistant content is not uniquely embedded for {source_id}")
        global_start = assistant_start + local_start
        global_end = assistant_start + local_end
        mask = [
            int(start < global_end and end > global_start)
            for start, end in offsets
        ]
        selected = [index for index, value in enumerate(mask) if value]
        if not selected:
            raise ValueError(f"Decision label has no token span for {source_id}")
        if any(labels[index] == -100 for index in selected):
            raise ValueError(f"Decision mask escapes assistant target for {source_id}")
        return mask

    def __getitem__(self, index):
        sample = self.samples[index]
        rendered = self._render(sample["conversations"])
        generation_prefix = self._render_generation_prefix(
            sample["conversations"], sample["source_id"]
        )
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded.input_ids)
        offsets = [tuple(pair) for pair in encoded.offset_mapping]
        if len(input_ids) > self.max_length:
            raise RuntimeError(
                f"Prepared sample {sample['source_id']} has {len(input_ids)} tokens, "
                f"exceeding max_length={self.max_length}; target truncation is forbidden"
            )
        first_target = self._aligned_prefix_length(
            rendered,
            generation_prefix,
            sample["conversations"][2]["content"],
            input_ids,
            offsets,
            sample["source_id"],
        )
        labels = self._generate_labels(input_ids, first_target)
        if all(label == -100 for label in labels):
            raise RuntimeError(
                f"Prepared sample {sample['source_id']} contains no assistant target"
            )
        decision_mask = self._decision_mask(
            rendered,
            sample["conversations"],
            offsets,
            labels,
            sample["task_type"],
            sample["source_id"],
        )
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        labels += [-100] * padding
        decision_mask += [0] * padding
        attention_mask += [0] * padding
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(decision_mask, dtype=torch.bool),
            torch.tensor(sample["task_id"], dtype=torch.long),
            torch.tensor(sample["soft_target"], dtype=torch.float32),
            torch.tensor(attention_mask, dtype=torch.long),
        )


__all__ = ["TARGET_CONTRACT_ID", "VGMTStructuredDecisionDatasetV43"]
