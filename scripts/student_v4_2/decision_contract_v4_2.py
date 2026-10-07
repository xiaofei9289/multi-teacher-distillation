"""Pure decision-label parsing contract for the V4.2 student objective."""

from __future__ import annotations

import re


LABELS_BY_TASK = {
    "claim_verification": {"Supported", "Refuted", "NotEnoughInfo"},
    "query_decision": {"Answer", "Abstain"},
}

DECISION_RE = re.compile(
    r"\ADecision: (?P<label>Supported|Refuted|NotEnoughInfo|Answer|Abstain)\n"
)
ANY_DECISION_LINE_RE = re.compile(r"(?m)^Decision\s*:")


def decision_label_span(assistant_content: str, task_type: str) -> tuple[str, int, int]:
    """Return the unique first-line class label and its character span."""

    if task_type not in LABELS_BY_TASK:
        raise ValueError(f"Unsupported task type: {task_type}")
    match = DECISION_RE.match(assistant_content)
    if match is None or len(ANY_DECISION_LINE_RE.findall(assistant_content)) != 1:
        raise ValueError("Assistant target must have exactly one first-line Decision label")
    label = match.group("label")
    if label not in LABELS_BY_TASK[task_type]:
        raise ValueError(f"Decision {label} is incompatible with task {task_type}")
    start, end = match.span("label")
    return label, start, end
