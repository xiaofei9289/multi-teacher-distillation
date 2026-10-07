"""Joint generation and hierarchical selective-decision losses for V5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .hierarchical_head_v5 import CLAIM_TASK_ID, QUERY_TASK_ID, HierarchicalDecisionOutput


LOSS_CONTRACT_ID = "hierarchical_selective_decoupled_soft_bce_v5"


def exclude_decision_span_from_lm_labels(
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor) or labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if not isinstance(decision_mask, torch.Tensor) or decision_mask.shape != labels.shape:
        raise ValueError("decision_mask must match labels")
    mask = decision_mask.to(device=labels.device, dtype=torch.bool)
    if bool((~mask.any(dim=1)).any()) or bool((mask & labels.eq(ignore_index)).any()):
        raise ValueError("each Decision mask must cover only supervised tokens")
    result = labels.clone()
    result.masked_fill_(mask, ignore_index)
    return result


@dataclass(frozen=True)
class GenerationLoss:
    loss: torch.Tensor
    valid_token_count: int
    excluded_decision_token_count: int


def compute_decoupled_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> GenerationLoss:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits and labels must align as [batch, sequence, vocabulary]")
    decoupled = exclude_decision_span_from_lm_labels(labels, decision_mask, ignore_index=ignore_index)
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = decoupled[:, 1:].contiguous()
    valid = shifted_labels.ne(ignore_index)
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("no generation target remains after Decision exclusion")
    loss = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    ) / count
    return GenerationLoss(loss, count, int(decision_mask[:, 1:].bool().sum().item()))


def binary_soft_target_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    row_weights: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("binary logits must have shape [rows]")
    if not isinstance(targets, torch.Tensor) or targets.shape != logits.shape:
        raise ValueError("binary targets must match logits")
    work_dtype = torch.float64 if targets.dtype == torch.float64 else torch.float32
    work_logits = logits.to(dtype=work_dtype)
    targets = targets.to(device=logits.device, dtype=work_dtype)
    if bool((~torch.isfinite(targets)).any()) or bool(((targets < 0) | (targets > 1)).any()):
        raise ValueError("binary soft targets must lie in [0, 1]")
    pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=work_dtype)
    if not bool(torch.isfinite(pos_weight)) or bool(pos_weight <= 0):
        raise ValueError("positive_weight must be finite and positive")
    losses = -(pos_weight * targets * F.logsigmoid(work_logits) + (1.0 - targets) * F.logsigmoid(-work_logits))
    if row_weights is not None:
        if not isinstance(row_weights, torch.Tensor) or row_weights.shape != logits.shape:
            raise ValueError("row_weights must match logits")
        weights = row_weights.to(device=logits.device, dtype=work_dtype)
        if bool((~torch.isfinite(weights)).any()) or bool((weights < 0).any()):
            raise ValueError("row_weights must be finite and non-negative")
        losses = losses * weights
        denominator = weights.sum()
    else:
        denominator = torch.as_tensor(logits.numel(), device=logits.device, dtype=work_dtype)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        if bool(denominator <= 0):
            raise ValueError("mean loss has zero effective target mass")
        return losses.sum() / denominator
    raise ValueError("reduction must be none, sum, or mean")


@dataclass(frozen=True)
class HierarchicalDecisionLoss:
    loss: torch.Tensor
    sufficiency_loss: torch.Tensor | None
    polarity_loss: torch.Tensor | None
    answerability_loss: torch.Tensor | None
    claim_count: int
    query_count: int
    polarity_effective_mass: float


def compute_hierarchical_decision_loss(
    decisions: HierarchicalDecisionOutput,
    sufficiency_targets: torch.Tensor,
    polarity_targets: torch.Tensor,
    polarity_weights: torch.Tensor,
    answerability_targets: torch.Tensor,
    *,
    sufficiency_positive_weight: float = 1.0,
    polarity_positive_weight: float = 1.0,
    answerability_positive_weight: float = 1.0,
    sufficiency_loss_weight: float = 1.0,
    polarity_loss_weight: float = 1.0,
    answerability_loss_weight: float = 1.0,
) -> HierarchicalDecisionLoss:
    batch = decisions.task_ids.shape[0]
    for name, target in (
        ("sufficiency_targets", sufficiency_targets),
        ("polarity_targets", polarity_targets),
        ("polarity_weights", polarity_weights),
        ("answerability_targets", answerability_targets),
    ):
        if not isinstance(target, torch.Tensor) or target.shape != (batch,):
            raise ValueError(f"{name} must have shape [batch]")
    claim = decisions.task_ids.eq(CLAIM_TASK_ID)
    query = decisions.task_ids.eq(QUERY_TASK_ID)
    claim_count, query_count = int(claim.sum().item()), int(query.sum().item())
    if claim_count + query_count != batch or batch == 0:
        raise ValueError("invalid or empty task assignment")
    components = []
    sufficiency_loss = None
    polarity_loss = None
    answerability_loss = None
    polarity_mass = 0.0
    if claim_count:
        sufficiency_loss = binary_soft_target_bce(
            decisions.sufficiency_logits[claim],
            sufficiency_targets[claim],
            positive_weight=sufficiency_positive_weight,
        )
        components.append(float(sufficiency_loss_weight) * sufficiency_loss)
        weights = polarity_weights[claim]
        polarity_mass = float(weights.sum().item())
        if polarity_mass > 0:
            polarity_loss = binary_soft_target_bce(
                decisions.polarity_logits[claim],
                polarity_targets[claim],
                positive_weight=polarity_positive_weight,
                row_weights=weights,
            )
            components.append(float(polarity_loss_weight) * polarity_loss)
    if query_count:
        answerability_loss = binary_soft_target_bce(
            decisions.answerability_logits[query],
            answerability_targets[query],
            positive_weight=answerability_positive_weight,
        )
        components.append(float(answerability_loss_weight) * answerability_loss)
    loss = sum(components, decisions.sufficiency_logits.new_zeros(()))
    if not bool(torch.isfinite(loss)):
        raise ValueError("non-finite hierarchical decision loss")
    return HierarchicalDecisionLoss(
        loss=loss,
        sufficiency_loss=sufficiency_loss,
        polarity_loss=polarity_loss,
        answerability_loss=answerability_loss,
        claim_count=claim_count,
        query_count=query_count,
        polarity_effective_mass=polarity_mass,
    )


@dataclass(frozen=True)
class V5JointLoss:
    total_loss: torch.Tensor
    generation_loss: torch.Tensor
    decision_loss: torch.Tensor
    aux_loss: torch.Tensor
    decision_components: HierarchicalDecisionLoss
    generation_valid_token_count: int
    excluded_decision_token_count: int


def compute_v5_joint_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    decision_mask: torch.Tensor,
    decisions: HierarchicalDecisionOutput,
    sufficiency_targets: torch.Tensor,
    polarity_targets: torch.Tensor,
    polarity_weights: torch.Tensor,
    answerability_targets: torch.Tensor,
    *,
    generation_loss_weight: float = 1.0,
    sufficiency_loss_weight: float = 1.0,
    polarity_loss_weight: float = 1.0,
    answerability_loss_weight: float = 1.0,
    sufficiency_positive_weight: float = 1.0,
    polarity_positive_weight: float = 1.0,
    answerability_positive_weight: float = 1.0,
    aux_loss: torch.Tensor | None = None,
) -> V5JointLoss:
    generation = compute_decoupled_causal_lm_loss(lm_logits, labels, decision_mask)
    decision = compute_hierarchical_decision_loss(
        decisions,
        sufficiency_targets,
        polarity_targets,
        polarity_weights,
        answerability_targets,
        sufficiency_positive_weight=sufficiency_positive_weight,
        polarity_positive_weight=polarity_positive_weight,
        answerability_positive_weight=answerability_positive_weight,
        sufficiency_loss_weight=sufficiency_loss_weight,
        polarity_loss_weight=polarity_loss_weight,
        answerability_loss_weight=answerability_loss_weight,
    )
    auxiliary = lm_logits.new_zeros(()) if aux_loss is None else aux_loss.to(lm_logits.device).reshape(())
    total = float(generation_loss_weight) * generation.loss + decision.loss + auxiliary
    if not bool(torch.isfinite(total)):
        raise ValueError("non-finite V5 joint loss")
    return V5JointLoss(
        total_loss=total,
        generation_loss=generation.loss,
        decision_loss=decision.loss,
        aux_loss=auxiliary,
        decision_components=decision,
        generation_valid_token_count=generation.valid_token_count,
        excluded_decision_token_count=generation.excluded_decision_token_count,
    )
