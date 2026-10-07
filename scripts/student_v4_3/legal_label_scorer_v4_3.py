"""Zero-training, task-constrained sequence scoring for V4.3 decisions.

The scorer enumerates only the legal decision labels for the requested task.
For each label it appends ``Decision: <label>\n`` to an already rendered
assistant-generation prefix and computes the mean causal log-probability of
the tokenizer pieces that overlap ``<label>``.  It has no trainable or fitted
state and deliberately accepts neither gold labels nor evaluation rows.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch


# Tuples freeze both the legal surface forms and deterministic tie order.
LEGAL_LABELS_BY_TASK: Mapping[str, tuple[str, ...]] = {
    "claim_verification": ("Supported", "Refuted", "NotEnoughInfo"),
    "query_decision": ("Answer", "Abstain"),
}

DECISION_PREFIX = "Decision: "
LABEL_TERMINATOR = "\n"
SCORING_RULE = "mean_causal_log_probability_over_label_overlap_tokens"


@dataclass(frozen=True)
class CandidateEncoding:
    """Tokenized candidate and the positions whose tokens encode its label."""

    label: str
    input_ids: tuple[int, ...]
    label_token_positions: tuple[int, ...]
    label_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LabelSequenceScore:
    """Auditable causal sequence score for one legal label."""

    label: str
    token_ids: tuple[int, ...]
    token_log_probabilities: tuple[float, ...]
    token_count: int
    log_likelihood_sum: float
    normalized_log_likelihood: float


@dataclass(frozen=True)
class LegalLabelScoringResult:
    """Task-masked prediction and all scores needed for an audit trail."""

    task_type: str
    prediction: str
    margin: float
    scoring_rule: str
    decision_prefix: str
    label_terminator: str
    scores: tuple[LabelSequenceScore, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def legal_labels_for_task(task_type: str) -> tuple[str, ...]:
    """Return the frozen legal labels for ``task_type`` or fail closed."""

    try:
        return LEGAL_LABELS_BY_TASK[task_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported task type: {task_type}") from exc


def _encoded_field(encoded: Any, name: str) -> Any:
    if isinstance(encoded, Mapping):
        if name not in encoded:
            raise ValueError(f"Tokenizer output is missing {name}")
        return encoded[name]
    if not hasattr(encoded, name):
        raise ValueError(f"Tokenizer output is missing {name}")
    return getattr(encoded, name)


def _flatten_single_encoding(value: Any, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    else:
        value = list(value)
    first_is_offset_pair = (
        name == "offset_mapping"
        and value
        and isinstance(value[0], (list, tuple))
        and len(value[0]) == 2
        and all(not isinstance(item, (list, tuple)) for item in value[0])
    )
    if value and isinstance(value[0], (list, tuple)) and not first_is_offset_pair:
        if len(value) != 1:
            raise ValueError(f"Expected one tokenizer sequence for {name}")
        value = list(value[0])
    return value


def encode_label_candidate(
    tokenizer: Any,
    generation_prefix: str,
    label: str,
    *,
    decision_prefix: str = DECISION_PREFIX,
    label_terminator: str = LABEL_TERMINATOR,
) -> CandidateEncoding:
    """Tokenize one candidate and identify every token overlapping its label.

    Offsets are computed on the complete candidate so tokenizer merges at the
    prefix/label and label/newline boundaries match real causal-LM inference.
    A boundary-spanning token is counted exactly once.
    """

    if not isinstance(generation_prefix, str) or not generation_prefix:
        raise ValueError("generation_prefix must be a non-empty rendered string")
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if not isinstance(decision_prefix, str) or not decision_prefix:
        raise ValueError("decision_prefix must be a non-empty string")
    if not isinstance(label_terminator, str) or not label_terminator:
        raise ValueError("label_terminator must be a non-empty string")

    label_start = len(generation_prefix) + len(decision_prefix)
    label_end = label_start + len(label)
    candidate_text = (
        generation_prefix + decision_prefix + label + label_terminator
    )
    encoded = tokenizer(
        candidate_text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    input_ids = [
        int(value)
        for value in _flatten_single_encoding(
            _encoded_field(encoded, "input_ids"), "input_ids"
        )
    ]
    raw_offsets = _flatten_single_encoding(
        _encoded_field(encoded, "offset_mapping"), "offset_mapping"
    )
    offsets: list[tuple[int, int]] = []
    for offset in raw_offsets:
        if isinstance(offset, torch.Tensor):
            offset = offset.detach().cpu().tolist()
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("Tokenizer offset_mapping must contain (start, end) pairs")
        start, end = int(offset[0]), int(offset[1])
        if start < 0 or end < start or end > len(candidate_text):
            raise ValueError(f"Invalid tokenizer offset: {(start, end)}")
        offsets.append((start, end))
    if len(input_ids) != len(offsets):
        raise ValueError("Tokenizer input_ids and offset_mapping lengths differ")

    positions = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if start < label_end and end > label_start
    )
    if not positions:
        raise ValueError(f"Tokenizer produced no tokens for decision label {label!r}")
    if positions[0] == 0:
        raise ValueError("The first label token has no causal predecessor")

    covered_characters: set[int] = set()
    for position in positions:
        start, end = offsets[position]
        covered_characters.update(
            range(max(start, label_start), min(end, label_end))
        )
    expected_characters = set(range(label_start, label_end))
    if covered_characters != expected_characters:
        raise ValueError(
            f"Tokenizer offsets do not fully cover decision label {label!r}"
        )

    return CandidateEncoding(
        label=label,
        input_ids=tuple(input_ids),
        label_token_positions=positions,
        label_token_ids=tuple(input_ids[position] for position in positions),
    )


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _extract_logits(output: Any) -> torch.Tensor:
    if hasattr(output, "logits"):
        logits = output.logits
    elif isinstance(output, Mapping) and "logits" in output:
        logits = output["logits"]
    elif isinstance(output, (list, tuple)) and output:
        logits = output[0]
    else:
        raise ValueError("Model output does not expose logits")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("Model logits must be a [batch, sequence, vocabulary] tensor")
    return logits


def _padding_token_id(tokenizer: Any) -> int:
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise ValueError("Tokenizer must define pad_token_id or eos_token_id")


def score_legal_label_sequences(
    model: Any,
    tokenizer: Any,
    generation_prefix: str,
    task_type: str,
    *,
    device: str | torch.device | None = None,
    decision_prefix: str = DECISION_PREFIX,
    label_terminator: str = LABEL_TERMINATOR,
) -> LegalLabelScoringResult:
    """Score every and only legal label, returning the mean-logprob argmax.

    The forward pass is deterministic evaluation only.  The model's original
    training/evaluation state is restored before returning, including on error.
    """

    labels = legal_labels_for_task(task_type)
    candidates = tuple(
        encode_label_candidate(
            tokenizer,
            generation_prefix,
            label,
            decision_prefix=decision_prefix,
            label_terminator=label_terminator,
        )
        for label in labels
    )
    target_device = torch.device(device) if device is not None else _model_device(model)
    maximum_length = max(len(candidate.input_ids) for candidate in candidates)
    batch_size = len(candidates)
    input_ids = torch.full(
        (batch_size, maximum_length),
        _padding_token_id(tokenizer),
        dtype=torch.long,
        device=target_device,
    )
    attention_mask = torch.zeros(
        (batch_size, maximum_length),
        dtype=torch.long,
        device=target_device,
    )
    for row_index, candidate in enumerate(candidates):
        sequence_length = len(candidate.input_ids)
        input_ids[row_index, :sequence_length] = torch.tensor(
            candidate.input_ids, dtype=torch.long, device=target_device
        )
        attention_mask[row_index, :sequence_length] = 1

    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = _extract_logits(output)
            if logits.shape[0] != batch_size or logits.shape[1] < maximum_length:
                raise ValueError(
                    "Model logits do not cover every candidate input position"
                )
            log_probabilities = torch.log_softmax(logits.float(), dim=-1)
    finally:
        if hasattr(model, "train"):
            model.train(was_training)

    scores: list[LabelSequenceScore] = []
    vocabulary_size = int(log_probabilities.shape[-1])
    for row_index, candidate in enumerate(candidates):
        token_log_probabilities: list[float] = []
        for position, token_id in zip(
            candidate.label_token_positions, candidate.label_token_ids
        ):
            if token_id < 0 or token_id >= vocabulary_size:
                raise ValueError(
                    f"Token id {token_id} lies outside model vocabulary size "
                    f"{vocabulary_size}"
                )
            value = float(log_probabilities[row_index, position - 1, token_id].item())
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite log-probability for label {candidate.label!r}"
                )
            token_log_probabilities.append(value)
        score_sum = math.fsum(token_log_probabilities)
        score_mean = score_sum / len(token_log_probabilities)
        scores.append(
            LabelSequenceScore(
                label=candidate.label,
                token_ids=candidate.label_token_ids,
                token_log_probabilities=tuple(token_log_probabilities),
                token_count=len(token_log_probabilities),
                log_likelihood_sum=score_sum,
                normalized_log_likelihood=score_mean,
            )
        )

    # Python's stable sort makes the declared legal-label order the exact tie rule.
    ranked = sorted(
        enumerate(scores),
        key=lambda item: (-item[1].normalized_log_likelihood, item[0]),
    )
    prediction = ranked[0][1].label
    margin = (
        ranked[0][1].normalized_log_likelihood
        - ranked[1][1].normalized_log_likelihood
    )
    return LegalLabelScoringResult(
        task_type=task_type,
        prediction=prediction,
        margin=margin,
        scoring_rule=SCORING_RULE,
        decision_prefix=decision_prefix,
        label_terminator=label_terminator,
        scores=tuple(scores),
    )


__all__ = [
    "CandidateEncoding",
    "DECISION_PREFIX",
    "LABEL_TERMINATOR",
    "LEGAL_LABELS_BY_TASK",
    "LegalLabelScoringResult",
    "LabelSequenceScore",
    "SCORING_RULE",
    "encode_label_candidate",
    "legal_labels_for_task",
    "score_legal_label_sequences",
]
