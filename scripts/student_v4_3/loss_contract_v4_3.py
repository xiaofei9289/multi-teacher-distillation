"""Pure V4.3 losses for decision--generation decoupling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .decision_head_v4_3 import (
    CLAIM_TASK_ID,
    QUERY_TASK_ID,
    TaskMaskedDecisionOutput,
)


LOSS_CONTRACT_ID = "task_head_decoupled_soft_ce_v4_3"


def exclude_decision_span_from_lm_labels(
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Clone labels and remove Decision-label tokens from generation targets."""

    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if not isinstance(decision_mask, torch.Tensor) or decision_mask.shape != labels.shape:
        raise ValueError("decision_mask must have the same shape as labels")
    decision_mask = decision_mask.to(device=labels.device, dtype=torch.bool)
    if bool((~decision_mask.any(dim=1)).any()):
        raise ValueError("every row must mark at least one Decision-label token")
    if bool((decision_mask & labels.eq(ignore_index)).any()):
        raise ValueError("decision_mask covers an already-unsupervised token")
    decoupled = labels.clone()
    decoupled.masked_fill_(decision_mask, ignore_index)
    return decoupled


@dataclass(frozen=True)
class DecoupledCausalLoss:
    loss: torch.Tensor
    valid_token_count: int
    excluded_decision_token_count: int


def compute_decoupled_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> DecoupledCausalLoss:
    """Compute causal LM CE after excluding the Decision value span."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must align with logits batch and sequence dimensions")
    if logits.shape[1] < 2:
        raise ValueError("causal loss requires at least two sequence positions")
    decoupled = exclude_decision_span_from_lm_labels(
        labels,
        decision_mask,
        ignore_index=ignore_index,
    )
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = decoupled[..., 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)
    valid_count = int(valid.sum().item())
    excluded_count = int(decision_mask[..., 1:].to(torch.bool).sum().item())
    if valid_count == 0:
        raise ValueError("no generation target remains after Decision exclusion")
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    ) / valid_count
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite decoupled causal LM loss")
    return DecoupledCausalLoss(loss, valid_count, excluded_count)


def class_balanced_weights_from_counts(
    counts: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return inverse-frequency ``N/(K*n_c)`` weights with arithmetic mean one."""

    if not isinstance(counts, torch.Tensor) or counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("counts must be a rank-1 tensor with at least two classes")
    numeric = counts.to(dtype=dtype)
    if not numeric.is_floating_point():
        raise ValueError("dtype must be floating point")
    if not bool(torch.isfinite(numeric).all()) or bool((numeric <= 0).any()):
        raise ValueError("all class counts must be finite and strictly positive")
    raw = numeric.sum() / (numeric.numel() * numeric)
    weights = raw / raw.mean()
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("non-finite class-balanced weights")
    return weights


def soft_target_class_balanced_cross_entropy(
    logits: torch.Tensor,
    soft_targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy for probability targets with optional per-class weighting."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if not isinstance(soft_targets, torch.Tensor) or soft_targets.shape != logits.shape:
        raise ValueError("soft_targets must have the same shape as logits")
    if logits.shape[0] == 0:
        raise ValueError("soft-target CE requires at least one example")
    work_dtype = (
        torch.float32
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits.dtype
    )
    work_logits = logits.to(dtype=work_dtype)
    targets = soft_targets.to(device=logits.device, dtype=work_dtype)
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError("soft_targets must be finite and non-negative")
    target_mass = targets.sum(dim=-1)
    if not bool(torch.allclose(target_mass, torch.ones_like(target_mass), atol=1e-5, rtol=1e-5)):
        raise ValueError("each soft-target row must sum to one")
    if class_weights is None:
        weights = torch.ones(logits.shape[1], dtype=work_dtype, device=logits.device)
    else:
        if not isinstance(class_weights, torch.Tensor) or class_weights.shape != (logits.shape[1],):
            raise ValueError("class_weights must have shape [classes]")
        weights = class_weights.to(device=logits.device, dtype=work_dtype)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
            raise ValueError("class_weights must be finite and strictly positive")
    per_example = -(targets * weights * F.log_softmax(work_logits, dim=-1)).sum(dim=-1)
    if reduction == "none":
        return per_example
    if reduction == "sum":
        return per_example.sum()
    if reduction == "mean":
        return per_example.mean()
    raise ValueError("reduction must be one of: none, sum, mean")


@dataclass(frozen=True)
class TaskMaskedDecisionLoss:
    loss: torch.Tensor
    claim_loss: torch.Tensor | None
    query_loss: torch.Tensor | None
    claim_count: int
    query_count: int


def compute_task_masked_decision_loss(
    decisions: TaskMaskedDecisionOutput,
    padded_soft_targets: torch.Tensor,
    *,
    claim_class_weights: torch.Tensor | None = None,
    query_class_weights: torch.Tensor | None = None,
) -> TaskMaskedDecisionLoss:
    """Normalize legal-label CE by the batch's summed weighted target mass."""

    batch_size = decisions.task_ids.shape[0]
    if not isinstance(padded_soft_targets, torch.Tensor) or padded_soft_targets.shape != (
        batch_size,
        3,
    ):
        raise ValueError("padded_soft_targets must have shape [batch, 3]")
    if not padded_soft_targets.is_floating_point():
        raise ValueError("padded_soft_targets must use a floating-point dtype")
    # Soft targets are supervision, not model activations.  Never downcast the
    # frozen fp32 q vectors merely because autocast produced fp16 head logits.
    # Float64 remains float64 for exact synthetic/reference tests; every lower
    # precision input is promoted to fp32.
    target_work_dtype = (
        torch.float64
        if padded_soft_targets.dtype == torch.float64
        else torch.float32
    )
    targets = padded_soft_targets.to(
        device=decisions.claim_logits.device,
        dtype=target_work_dtype,
    )
    if not bool(torch.isfinite(targets).all()) or bool((targets < 0).any()):
        raise ValueError("padded_soft_targets must be finite and non-negative")

    claim_rows = decisions.task_ids.eq(CLAIM_TASK_ID)
    query_rows = decisions.task_ids.eq(QUERY_TASK_ID)
    claim_count = int(claim_rows.sum().item())
    query_count = int(query_rows.sum().item())
    if claim_count + query_count != batch_size or batch_size == 0:
        raise ValueError("decision output contains invalid or empty task assignments")
    if query_count and bool((targets[query_rows, 2] != 0).any()):
        raise ValueError("query targets may not place mass on the padded third class")

    total = decisions.claim_logits.new_zeros((), dtype=torch.float32)
    total_weighted_mass = decisions.claim_logits.new_zeros((), dtype=torch.float32)
    claim_loss = None
    query_loss = None
    if claim_count:
        claim_per_example = soft_target_class_balanced_cross_entropy(
            decisions.claim_logits[claim_rows],
            targets[claim_rows],
            claim_class_weights,
            reduction="none",
        )
        claim_loss = claim_per_example.mean()
        total = total + claim_per_example.sum()
        claim_targets = targets[claim_rows].float()
        claim_weights = (
            torch.ones(3, device=targets.device, dtype=torch.float32)
            if claim_class_weights is None
            else claim_class_weights.to(device=targets.device, dtype=torch.float32)
        )
        total_weighted_mass = total_weighted_mass + (
            claim_targets * claim_weights
        ).sum()
    if query_count:
        query_per_example = soft_target_class_balanced_cross_entropy(
            decisions.query_logits[query_rows],
            targets[query_rows, :2],
            query_class_weights,
            reduction="none",
        )
        query_loss = query_per_example.mean()
        total = total + query_per_example.sum()
        query_targets = targets[query_rows, :2].float()
        query_weights = (
            torch.ones(2, device=targets.device, dtype=torch.float32)
            if query_class_weights is None
            else query_class_weights.to(device=targets.device, dtype=torch.float32)
        )
        total_weighted_mass = total_weighted_mass + (
            query_targets * query_weights
        ).sum()
    if not bool(torch.isfinite(total_weighted_mass)) or bool(total_weighted_mass <= 0):
        raise ValueError("non-positive weighted soft-target mass")
    loss = total / total_weighted_mass
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite task-masked decision loss")
    return TaskMaskedDecisionLoss(loss, claim_loss, query_loss, claim_count, query_count)


@dataclass(frozen=True)
class V43JointLoss:
    total_loss: torch.Tensor
    generation_loss: torch.Tensor
    decision_loss: torch.Tensor
    aux_loss: torch.Tensor
    generation_valid_token_count: int
    excluded_decision_token_count: int
    claim_count: int
    query_count: int


def compute_v43_joint_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    decisions: TaskMaskedDecisionOutput,
    padded_soft_targets: torch.Tensor,
    *,
    claim_class_weights: torch.Tensor | None = None,
    query_class_weights: torch.Tensor | None = None,
    decision_loss_weight: float = 1.0,
    aux_loss: torch.Tensor | None = None,
) -> V43JointLoss:
    """Return ``L_generation + beta * L_decision + L_aux`` and audit counts."""

    beta = torch.as_tensor(
        decision_loss_weight,
        device=lm_logits.device,
        dtype=lm_logits.dtype,
    )
    if beta.ndim != 0 or not bool(torch.isfinite(beta)) or bool(beta < 0):
        raise ValueError("decision_loss_weight must be a finite non-negative scalar")
    generation = compute_decoupled_causal_lm_loss(
        lm_logits,
        labels,
        decision_mask,
    )
    decision = compute_task_masked_decision_loss(
        decisions,
        padded_soft_targets,
        claim_class_weights=claim_class_weights,
        query_class_weights=query_class_weights,
    )
    if aux_loss is None:
        auxiliary = lm_logits.new_zeros(())
    elif not isinstance(aux_loss, torch.Tensor) or aux_loss.numel() != 1:
        raise ValueError("aux_loss must be a scalar tensor")
    else:
        auxiliary = aux_loss.to(device=lm_logits.device, dtype=lm_logits.dtype).reshape(())
    total = generation.loss + beta * decision.loss + auxiliary
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite V4.3 joint loss")
    return V43JointLoss(
        total_loss=total,
        generation_loss=generation.loss,
        decision_loss=decision.loss,
        aux_loss=auxiliary,
        generation_valid_token_count=generation.valid_token_count,
        excluded_decision_token_count=generation.excluded_decision_token_count,
        claim_count=decision.claim_count,
        query_count=decision.query_count,
    )
