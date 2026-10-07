"""V4.3 student components for decision--generation decoupling."""

from .decision_head_v4_3 import (
    CLAIM_TASK_ID,
    QUERY_TASK_ID,
    MiniMindWithTaskDecisionHeads,
    TaskMaskedDecisionHeads,
    V43ModelOutput,
)
from .loss_contract_v4_3 import (
    LOSS_CONTRACT_ID,
    class_balanced_weights_from_counts,
    compute_decoupled_causal_lm_loss,
    compute_task_masked_decision_loss,
    compute_v43_joint_loss,
    exclude_decision_span_from_lm_labels,
    soft_target_class_balanced_cross_entropy,
)
from .legal_label_scorer_v4_3 import (
    LEGAL_LABELS_BY_TASK,
    LegalLabelScoringResult,
    encode_label_candidate,
    legal_labels_for_task,
    score_legal_label_sequences,
)

__all__ = [
    "CLAIM_TASK_ID",
    "QUERY_TASK_ID",
    "LOSS_CONTRACT_ID",
    "LEGAL_LABELS_BY_TASK",
    "LegalLabelScoringResult",
    "MiniMindWithTaskDecisionHeads",
    "TaskMaskedDecisionHeads",
    "V43ModelOutput",
    "class_balanced_weights_from_counts",
    "compute_decoupled_causal_lm_loss",
    "compute_task_masked_decision_loss",
    "compute_v43_joint_loss",
    "encode_label_candidate",
    "exclude_decision_span_from_lm_labels",
    "legal_labels_for_task",
    "score_legal_label_sequences",
    "soft_target_class_balanced_cross_entropy",
]
