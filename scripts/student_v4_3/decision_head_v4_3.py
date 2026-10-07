"""Task-masked V4.3 decision heads wrapped around an unchanged MiniMind LM.

The classifier reads the hidden state immediately before the first supervised
assistant token.  Under causal attention this state cannot contain the gold
Decision label or any later rationale/citation tokens.  The wrapped
``MiniMindForCausalLM`` instance is deliberately treated as an opaque base
model; its source and forward contract remain frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


CLAIM_TASK_ID = 0
QUERY_TASK_ID = 1
CLAIM_LABELS = ("Supported", "Refuted", "NotEnoughInfo")
QUERY_LABELS = ("Answer", "Abstain")
MAX_NUM_CLASSES = len(CLAIM_LABELS)


def task_local_argmax_with_margin(
    probabilities: torch.Tensor,
) -> tuple[int, float]:
    """Return first-max local index and its margin over all other classes."""

    if not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 1:
        raise ValueError("probabilities must be a rank-1 tensor")
    if probabilities.numel() < 2 or not probabilities.is_floating_point():
        raise ValueError("probabilities must contain at least two floating values")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("probabilities must be finite")
    selected = int(torch.argmax(probabilities).item())
    competitors = torch.cat(
        (probabilities[:selected], probabilities[selected + 1:])
    )
    margin = float((probabilities[selected] - competitors.max()).item())
    return selected, margin


def _validated_task_ids(
    task_ids: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(task_ids, torch.Tensor) or task_ids.ndim != 1:
        raise ValueError("task_ids must be a rank-1 tensor")
    if task_ids.shape[0] != batch_size:
        raise ValueError("task_ids batch dimension does not match hidden states")
    if task_ids.dtype == torch.bool or task_ids.is_floating_point():
        raise ValueError("task_ids must use an integer dtype")
    task_ids = task_ids.to(device=device, dtype=torch.long)
    if bool(((task_ids != CLAIM_TASK_ID) & (task_ids != QUERY_TASK_ID)).any()):
        raise ValueError("task_ids may contain only claim=0 or query=1")
    return task_ids


def infer_prompt_end_indices(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return the index just before each row's first supervised target token.

    ``labels`` follows the existing MiniMind SFT convention: prompt/template
    tokens are ``ignore_index`` and assistant target tokens retain token IDs.
    Selecting ``first_target - 1`` makes the decision representation blind to
    the teacher-forced label while preserving the complete prompt.
    """

    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if labels.shape[0] == 0 or labels.shape[1] < 2:
        raise ValueError("labels must contain a non-empty batch and sequence")
    supervised = labels.ne(ignore_index)
    if bool((~supervised.any(dim=1)).any()):
        raise ValueError("every row must contain at least one supervised target token")
    first_target = supervised.to(torch.long).argmax(dim=1)
    if bool((first_target == 0).any()):
        raise ValueError("a prompt-end state cannot precede a target at position zero")
    prompt_end = first_target - 1

    if attention_mask is not None:
        if attention_mask.shape != labels.shape:
            raise ValueError("attention_mask must have the same shape as labels")
        active = attention_mask.to(device=labels.device, dtype=torch.bool)
        if bool((supervised & ~active).any()):
            raise ValueError("a supervised target token is hidden by attention_mask")
        if bool((~active.gather(1, prompt_end[:, None]).squeeze(1)).any()):
            raise ValueError("the inferred prompt-end token is hidden by attention_mask")
    return prompt_end


def select_prompt_end_hidden(
    hidden_states: torch.Tensor,
    prompt_end_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather one leakage-safe prompt representation from each batch row."""

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    batch_size, sequence_length, _ = hidden_states.shape
    if not isinstance(prompt_end_indices, torch.Tensor) or prompt_end_indices.ndim != 1:
        raise ValueError("prompt_end_indices must be a rank-1 tensor")
    if prompt_end_indices.shape[0] != batch_size:
        raise ValueError("prompt_end_indices batch dimension does not match hidden states")
    if prompt_end_indices.dtype == torch.bool or prompt_end_indices.is_floating_point():
        raise ValueError("prompt_end_indices must use an integer dtype")
    indices = prompt_end_indices.to(device=hidden_states.device, dtype=torch.long)
    if bool(((indices < 0) | (indices >= sequence_length)).any()):
        raise ValueError("prompt_end_indices contains an out-of-range position")
    batch = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch, indices]


@dataclass(frozen=True)
class TaskMaskedDecisionOutput:
    """Raw per-task logits plus a common masked view for legal-label decoding."""

    claim_logits: torch.Tensor
    query_logits: torch.Tensor
    task_ids: torch.Tensor
    pooled_hidden: torch.Tensor

    def __post_init__(self) -> None:
        batch_size = self.task_ids.shape[0]
        if self.claim_logits.shape != (batch_size, len(CLAIM_LABELS)):
            raise ValueError("claim_logits must have shape [batch, 3]")
        if self.query_logits.shape != (batch_size, len(QUERY_LABELS)):
            raise ValueError("query_logits must have shape [batch, 2]")
        if self.pooled_hidden.ndim != 2 or self.pooled_hidden.shape[0] != batch_size:
            raise ValueError("pooled_hidden must have shape [batch, hidden]")
        tensors = (self.claim_logits, self.query_logits, self.pooled_hidden)
        if any(tensor.device != self.task_ids.device for tensor in tensors):
            raise ValueError("decision tensors and task_ids must share one device")
        if self.claim_logits.dtype != self.query_logits.dtype:
            raise ValueError("claim and query logits must share one dtype")

    @property
    def valid_class_mask(self) -> torch.Tensor:
        """Boolean mask over the common ``[batch, 3]`` label space."""

        claim_mask = torch.ones(
            (self.task_ids.shape[0], MAX_NUM_CLASSES),
            dtype=torch.bool,
            device=self.task_ids.device,
        )
        query_mask = claim_mask.clone()
        query_mask[:, -1] = False
        return torch.where(self.task_ids[:, None] == CLAIM_TASK_ID, claim_mask, query_mask)

    @property
    def padded_logits(self) -> torch.Tensor:
        """Return logits with the query-only third class made unselectable."""

        if not self.claim_logits.is_floating_point() or not self.query_logits.is_floating_point():
            raise ValueError("decision logits must use a floating dtype")
        invalid = float("-inf")
        query_padded = torch.nn.functional.pad(
            self.query_logits,
            (0, MAX_NUM_CLASSES - len(QUERY_LABELS)),
            value=invalid,
        )
        return torch.where(
            self.task_ids[:, None] == CLAIM_TASK_ID,
            self.claim_logits,
            query_padded,
        )


class TaskMaskedDecisionHeads(nn.Module):
    """Independent three-way claim and two-way query classification heads."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.hidden_size = hidden_size
        self.claim_head = nn.Linear(hidden_size, len(CLAIM_LABELS), bias=True)
        self.query_head = nn.Linear(hidden_size, len(QUERY_LABELS), bias=True)
        # A neutral, deterministic start avoids importing a random class prior
        # into the very mechanism intended to diagnose category collapse.
        nn.init.zeros_(self.claim_head.weight)
        nn.init.zeros_(self.claim_head.bias)
        nn.init.zeros_(self.query_head.weight)
        nn.init.zeros_(self.query_head.bias)

    def forward(
        self,
        pooled_hidden: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> TaskMaskedDecisionOutput:
        if not isinstance(pooled_hidden, torch.Tensor) or pooled_hidden.ndim != 2:
            raise ValueError("pooled_hidden must have shape [batch, hidden]")
        if pooled_hidden.shape[1] != self.hidden_size:
            raise ValueError("pooled_hidden has the wrong hidden dimension")
        task_ids = _validated_task_ids(
            task_ids,
            batch_size=pooled_hidden.shape[0],
            device=pooled_hidden.device,
        )
        return TaskMaskedDecisionOutput(
            claim_logits=self.claim_head(pooled_hidden),
            query_logits=self.query_head(pooled_hidden),
            task_ids=task_ids,
            pooled_hidden=pooled_hidden,
        )


def _last_hidden_tensor(base_output: Any) -> torch.Tensor:
    hidden_states = getattr(base_output, "hidden_states", None)
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1] if hidden_states else None
    if hidden_states is None:
        hidden_states = getattr(base_output, "last_hidden_state", None)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError(
            "base MiniMind output must expose rank-3 hidden_states or last_hidden_state"
        )
    return hidden_states


@dataclass(frozen=True)
class V43ModelOutput:
    """Outputs needed by the separate generation and decision objectives."""

    base_output: Any
    decisions: TaskMaskedDecisionOutput
    prompt_end_indices: torch.Tensor

    @property
    def logits(self) -> torch.Tensor:
        logits = getattr(self.base_output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("base MiniMind output does not expose logits")
        return logits

    @property
    def aux_loss(self) -> torch.Tensor | None:
        return getattr(self.base_output, "aux_loss", None)


class MiniMindWithTaskDecisionHeads(nn.Module):
    """Non-invasive wrapper for an existing ``MiniMindForCausalLM`` instance."""

    def __init__(self, base_model: nn.Module, hidden_size: int | None = None) -> None:
        super().__init__()
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be a torch.nn.Module")
        inferred = getattr(getattr(base_model, "config", None), "hidden_size", None)
        hidden_size = inferred if hidden_size is None else hidden_size
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size is required when base_model.config lacks it")
        if inferred is not None and int(inferred) != hidden_size:
            raise ValueError("hidden_size disagrees with base_model.config.hidden_size")
        self.base_model = base_model
        self.decision_heads = TaskMaskedDecisionHeads(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        task_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        prompt_end_indices: torch.Tensor | None = None,
        use_cache: bool = False,
        **base_kwargs: Any,
    ) -> V43ModelOutput:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if labels is None and prompt_end_indices is None:
            raise ValueError("provide labels or explicit prompt_end_indices")
        inferred_indices = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            inferred_indices = infer_prompt_end_indices(labels, attention_mask)
        if prompt_end_indices is None:
            prompt_end_indices = inferred_indices
        elif inferred_indices is not None:
            provided = prompt_end_indices.to(
                device=inferred_indices.device,
                dtype=inferred_indices.dtype,
            )
            if provided.shape != inferred_indices.shape or not torch.equal(
                provided, inferred_indices
            ):
                raise ValueError("explicit prompt_end_indices disagrees with labels")
        prompt_end_indices = prompt_end_indices.to(
            device=input_ids.device,
            dtype=torch.long,
        )

        # Never pass labels to the base LM: its built-in loss still supervises
        # Decision tokens.  V4.3 computes the decoupled LM loss explicitly.
        base_output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **base_kwargs,
        )
        hidden_states = _last_hidden_tensor(base_output)
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("base hidden states do not align with input_ids")
        pooled_hidden = select_prompt_end_hidden(hidden_states, prompt_end_indices)
        decisions = self.decision_heads(pooled_hidden, task_ids)
        return V43ModelOutput(
            base_output=base_output,
            decisions=decisions,
            prompt_end_indices=prompt_end_indices,
        )
