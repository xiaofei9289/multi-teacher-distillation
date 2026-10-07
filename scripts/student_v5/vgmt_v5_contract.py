"""Pure contracts shared by the VG-MTD V5 data, gate, and student code."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping


CLAIM_LABELS = ("Supported", "Refuted", "NotEnoughInfo")
QUERY_LABELS = ("Answer", "Abstain")
ROLES = (
    "recommendation_teacher",
    "verification_teacher",
    "abstention_teacher",
)
TASKS = ("claim_verification", "query_decision")
HUMAN_LABEL_SOURCES = frozenset({"human_gold", "human_gold_v2"})


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_slug(model_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.casefold()).strip("-")
    if not slug:
        raise ValueError("model_name does not contain a usable identifier")
    return slug


def crossover_key(source_example_id: str, role: str, model_name: str) -> tuple[str, str, str]:
    if role not in ROLES:
        raise ValueError(f"Unknown teacher role: {role}")
    if not source_example_id or not model_name:
        raise ValueError("source_example_id and model_name are required")
    return source_example_id, role, model_name


def validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return value


def hierarchical_claim_distribution(p_suff: float, p_support: float) -> dict[str, float]:
    """Map factorized claim probabilities into the legal three-way simplex."""

    p_suff = validate_probability(p_suff, "p_suff")
    p_support = validate_probability(p_support, "p_support")
    distribution = {
        "Supported": p_suff * p_support,
        "Refuted": p_suff * (1.0 - p_support),
        "NotEnoughInfo": 1.0 - p_suff,
    }
    if not math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("hierarchical claim probabilities left the simplex")
    return distribution


def _nonnegative_mass(mass: Mapping[str, float], legal: tuple[str, ...]) -> dict[str, float]:
    unknown = set(mass) - set(legal)
    if unknown:
        raise ValueError(f"Illegal decision mass keys: {sorted(unknown)}")
    values = {label: float(mass.get(label, 0.0)) for label in legal}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("decision mass must be finite and non-negative")
    if sum(values.values()) <= 0.0:
        raise ValueError("decision mass must contain positive total weight")
    return values


def hierarchical_targets_from_mass(task_type: str, mass: Mapping[str, float]) -> dict:
    """Convert weighted teacher decision mass into V5 head targets.

    Claim polarity is neutral and masked when no sufficient teacher mass is
    available. Query rows expose only answerability.
    """

    if task_type == "claim_verification":
        values = _nonnegative_mass(mass, CLAIM_LABELS)
        sufficient = values["Supported"] + values["Refuted"]
        total = sufficient + values["NotEnoughInfo"]
        p_suff = sufficient / total
        polarity_mask = sufficient > 0.0
        p_support = values["Supported"] / sufficient if polarity_mask else 0.5
        return {
            "task_type": task_type,
            "p_sufficiency": p_suff,
            "p_support_given_sufficient": p_support,
            "polarity_loss_weight": p_suff if polarity_mask else 0.0,
            "p_answerable": None,
            "decision_distribution": hierarchical_claim_distribution(p_suff, p_support),
        }
    if task_type == "query_decision":
        values = _nonnegative_mass(mass, QUERY_LABELS)
        total = values["Answer"] + values["Abstain"]
        p_answerable = values["Answer"] / total
        return {
            "task_type": task_type,
            "p_sufficiency": None,
            "p_support_given_sufficient": None,
            "polarity_loss_weight": 0.0,
            "p_answerable": p_answerable,
            "decision_distribution": {
                "Answer": p_answerable,
                "Abstain": 1.0 - p_answerable,
            },
        }
    raise ValueError(f"Unknown task_type: {task_type}")


def task_labels(task_type: str) -> tuple[str, ...]:
    if task_type == "claim_verification":
        return CLAIM_LABELS
    if task_type == "query_decision":
        return QUERY_LABELS
    raise ValueError(f"Unknown task_type: {task_type}")


def is_valid_teacher_output(output_row: Mapping) -> bool:
    validation = output_row.get("validation", {})
    return all(
        bool(validation.get(field))
        for field in ("schema_valid", "task_decision_valid", "citation_valid")
    )
