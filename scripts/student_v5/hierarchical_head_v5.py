"""Leakage-safe hierarchical selective heads for the common V5 MiniMind student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


CLAIM_TASK_ID = 0
QUERY_TASK_ID = 1
CLAIM_LABELS = ("Supported", "Refuted", "NotEnoughInfo")
QUERY_LABELS = ("Answer", "Abstain")


def infer_prompt_end_indices(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    supervised = labels.ne(ignore_index)
    if labels.shape[0] == 0 or labels.shape[1] < 2 or bool((~supervised.any(dim=1)).any()):
        raise ValueError("every non-empty row must contain a supervised target")
    first_target = supervised.long().argmax(dim=1)
    if bool((first_target == 0).any()):
        raise ValueError("prompt-end state cannot precede position zero")
    prompt_end = first_target - 1
    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must match labels")
        active = attention_mask.to(device=labels.device, dtype=torch.bool)
        if bool((supervised & ~active).any()) or bool((~active.gather(1, prompt_end[:, None]).squeeze(1)).any()):
            raise ValueError("supervised or prompt-end tokens are hidden by attention_mask")
    return prompt_end


def select_prompt_end_hidden(hidden_states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1 or indices.shape[0] != hidden_states.shape[0]:
        raise ValueError("indices must contain one position per batch row")
    if indices.dtype == torch.bool or indices.is_floating_point():
        raise ValueError("indices must use an integer dtype")
    indices = indices.to(device=hidden_states.device, dtype=torch.long)
    if bool(((indices < 0) | (indices >= hidden_states.shape[1])).any()):
        raise ValueError("prompt-end index is out of range")
    batch = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch, indices]


def validated_task_ids(task_ids: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if not isinstance(task_ids, torch.Tensor) or task_ids.ndim != 1 or task_ids.shape[0] != batch_size:
        raise ValueError("task_ids must have shape [batch]")
    if task_ids.dtype == torch.bool or task_ids.is_floating_point():
        raise ValueError("task_ids must use an integer dtype")
    task_ids = task_ids.to(device=device, dtype=torch.long)
    if bool(((task_ids != CLAIM_TASK_ID) & (task_ids != QUERY_TASK_ID)).any()):
        raise ValueError("task_ids may contain only claim=0 and query=1")
    return task_ids


def hierarchical_claim_probabilities(
    sufficiency_logits: torch.Tensor,
    polarity_logits: torch.Tensor,
) -> torch.Tensor:
    if sufficiency_logits.shape != polarity_logits.shape:
        raise ValueError("sufficiency and polarity logits must have the same shape")
    p_sufficient = torch.sigmoid(sufficiency_logits.float())
    p_support = torch.sigmoid(polarity_logits.float())
    return torch.stack(
        (
            p_sufficient * p_support,
            p_sufficient * (1.0 - p_support),
            1.0 - p_sufficient,
        ),
        dim=-1,
    )


def query_probabilities(answerability_logits: torch.Tensor) -> torch.Tensor:
    p_answer = torch.sigmoid(answerability_logits.float())
    return torch.stack((p_answer, 1.0 - p_answer), dim=-1)


@dataclass(frozen=True)
class HierarchicalDecisionOutput:
    sufficiency_logits: torch.Tensor
    polarity_logits: torch.Tensor
    answerability_logits: torch.Tensor
    task_ids: torch.Tensor
    pooled_hidden: torch.Tensor

    def __post_init__(self) -> None:
        batch = self.task_ids.shape[0]
        for name, tensor in (
            ("sufficiency_logits", self.sufficiency_logits),
            ("polarity_logits", self.polarity_logits),
            ("answerability_logits", self.answerability_logits),
        ):
            if tensor.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if self.pooled_hidden.ndim != 2 or self.pooled_hidden.shape[0] != batch:
            raise ValueError("pooled_hidden must have shape [batch, hidden]")

    @property
    def claim_probabilities(self) -> torch.Tensor:
        return hierarchical_claim_probabilities(self.sufficiency_logits, self.polarity_logits)

    @property
    def query_probabilities(self) -> torch.Tensor:
        return query_probabilities(self.answerability_logits)

    @property
    def padded_task_probabilities(self) -> torch.Tensor:
        query_padded = torch.nn.functional.pad(self.query_probabilities, (0, 1), value=0.0)
        return torch.where(
            self.task_ids[:, None] == CLAIM_TASK_ID,
            self.claim_probabilities,
            query_padded,
        )


class HierarchicalSelectiveHeads(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.sufficiency_head = nn.Linear(hidden_size, 1)
        self.polarity_head = nn.Linear(hidden_size, 1)
        self.answerability_head = nn.Linear(hidden_size, 1)
        for layer in (self.sufficiency_head, self.polarity_head, self.answerability_head):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, pooled_hidden: torch.Tensor, task_ids: torch.Tensor) -> HierarchicalDecisionOutput:
        if not isinstance(pooled_hidden, torch.Tensor) or pooled_hidden.ndim != 2 or pooled_hidden.shape[1] != self.hidden_size:
            raise ValueError("pooled_hidden has an invalid shape")
        task_ids = validated_task_ids(task_ids, pooled_hidden.shape[0], pooled_hidden.device)
        return HierarchicalDecisionOutput(
            sufficiency_logits=self.sufficiency_head(pooled_hidden).squeeze(-1),
            polarity_logits=self.polarity_head(pooled_hidden).squeeze(-1),
            answerability_logits=self.answerability_head(pooled_hidden).squeeze(-1),
            task_ids=task_ids,
            pooled_hidden=pooled_hidden,
        )


def _last_hidden_tensor(base_output: Any) -> torch.Tensor:
    hidden = getattr(base_output, "hidden_states", None)
    if isinstance(hidden, (tuple, list)):
        hidden = hidden[-1] if hidden else None
    if hidden is None:
        hidden = getattr(base_output, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ValueError("base model must expose rank-3 hidden_states or last_hidden_state")
    return hidden


@dataclass(frozen=True)
class V5ModelOutput:
    base_output: Any
    decisions: HierarchicalDecisionOutput
    prompt_end_indices: torch.Tensor

    @property
    def logits(self) -> torch.Tensor:
        logits = getattr(self.base_output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("base model does not expose logits")
        return logits

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return getattr(self.base_output, "aux_loss", None)


class MiniMindWithHierarchicalHeads(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_size: int | None = None) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be a torch.nn.Module")
        inferred = getattr(getattr(base_model, "config", None), "hidden_size", None)
        hidden_size = inferred if hidden_size is None else hidden_size
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size is required")
        if inferred is not None and int(inferred) != hidden_size:
            raise ValueError("hidden_size disagrees with base model config")
        self.base_model = base_model
        self.decision_heads = HierarchicalSelectiveHeads(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_end_indices: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> V5ModelOutput:
        if prompt_end_indices is None:
            if labels is None:
                raise ValueError("labels or prompt_end_indices are required")
            prompt_end_indices = infer_prompt_end_indices(labels, attention_mask)
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        pooled = select_prompt_end_hidden(_last_hidden_tensor(base_output), prompt_end_indices)
        return V5ModelOutput(
            base_output=base_output,
            decisions=self.decision_heads(pooled, task_ids),
            prompt_end_indices=prompt_end_indices.to(input_ids.device),
        )
