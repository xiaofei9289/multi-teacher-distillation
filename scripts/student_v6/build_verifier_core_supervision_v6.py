#!/usr/bin/env python3
"""Build the budget-matched verifier-core V6 supervision matrix."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

try:
    from scripts.student_v5.fit_role_disentangled_gate_v5 import (
        feature_values,
        verifier_compatibility,
    )
    from scripts.student_v5.vgmt_v5_contract import (
        CLAIM_LABELS,
        ROLES,
        hierarchical_targets_from_mass,
        is_valid_teacher_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )
except ImportError:  # direct execution from this directory
    import sys

    ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
    from scripts.student_v5.fit_role_disentangled_gate_v5 import (  # type: ignore
        feature_values,
        verifier_compatibility,
    )
    from scripts.student_v5.vgmt_v5_contract import (  # type: ignore
        CLAIM_LABELS,
        ROLES,
        hierarchical_targets_from_mass,
        is_valid_teacher_output,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )


TARGET_CONTRACT_ID = "verifier_core_targets_v6"
SYSTEM_PROMPT = (
    "You are a small evidence-grounded recommendation model. Use only the "
    "provided evidence. If support is insufficient, conflicting, or stale, abstain."
)
FORMAL_VARIANTS = (
    "sft_only",
    "answer_only_kd",
    "single_teacher_rationale_kd",
    "unfiltered_mtd",
    "verifier_hard_filter_mtd",
    "verifier_weighted_mtd",
    "verifier_dynamic_product_mtd",
    "role_aware_verifier_weighted_mtd",
)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def predict_gate(feature_map: dict[str, float], model: dict) -> float:
    names = model["feature_names"]
    coefficients = model["coefficients"]
    if len(names) != len(coefficients):
        raise ValueError("Gate feature/coefficient length mismatch")
    missing = set(names) - set(feature_map)
    if missing:
        raise ValueError(f"Gate feature map is incomplete: {sorted(missing)}")
    logit = float(model["intercept"][0]) + sum(
        float(coefficient) * float(feature_map[name])
        for name, coefficient in zip(names, coefficients)
    )
    return sigmoid(logit)


def user_content(output: dict) -> str:
    item = output["input"]
    return "\n".join(
        [
            f"Task type: {item.get('task_type', '')}",
            f"Domain: {item.get('domain', '')}",
            f"Query: {item.get('query_text', '')}",
            f"Claim: {item.get('claim_text', '')}",
            f"Evidence: {item.get('evidence_text', '')}",
            f"Evidence id: {item.get('evidence_id', '')}",
            f"Review id: {item.get('review_id', '')}",
            f"Time bin: {item.get('time_bin', '')}",
        ]
    )


def materialized_decision(target: dict, contract: dict) -> str:
    minimum = float(contract["positive_min_probability"])
    margin = float(contract["safety_margin"])
    distribution = target["decision_distribution"]
    if target["task_type"] == "query_decision":
        p_answer = distribution["Answer"]
        return (
            "Answer"
            if p_answer >= minimum and p_answer - distribution["Abstain"] >= margin
            else "Abstain"
        )
    positive = max(
        ("Supported", "Refuted"),
        key=lambda label: (distribution[label], -CLAIM_LABELS.index(label)),
    )
    return (
        positive
        if distribution[positive] >= minimum
        and distribution[positive] - distribution["NotEnoughInfo"] >= margin
        else "NotEnoughInfo"
    )


def normalized(
    candidates: list[tuple[dict, float, str]],
) -> list[tuple[dict, float, str]]:
    total = sum(float(weight) for _, weight, _ in candidates)
    if total <= 0.0:
        raise ValueError("Candidate weights must contain positive mass")
    return [(row, float(weight) / total, source) for row, weight, source in candidates]


def _selected_by_gate(group: list[dict], gate_variant: str, threshold: float) -> list[dict]:
    selected = [
        row
        for row in group
        if is_valid_teacher_output(row["output"])
        and row["gate_probabilities"][gate_variant] >= threshold
    ]
    if selected:
        return selected
    valid = [row for row in group if is_valid_teacher_output(row["output"])]
    pool = valid or group
    return [
        max(
            pool,
            key=lambda row: (
                row["gate_probabilities"][gate_variant],
                row["request_id"],
            ),
        )
    ]


def candidate_set(
    variant: str,
    group: list[dict],
    gate_models: dict,
    minimum_weight: float,
) -> list[tuple[dict, float, str]]:
    by_role = {row["teacher_role"]: row for row in group}
    if variant == "sft_only":
        return []
    if variant in {"answer_only_kd", "single_teacher_rationale_kd"}:
        return [(by_role["recommendation_teacher"], 1.0, "fixed_recommendation")]
    valid = [row for row in group if is_valid_teacher_output(row["output"])]
    if variant == "unfiltered_mtd":
        pool = valid or group
        return normalized([(row, 1.0, "uniform") for row in pool])
    if variant in {
        "verifier_hard_filter_mtd",
        "verifier_weighted_mtd",
        "verifier_dynamic_product_mtd",
    }:
        gate_variant = "no_role"
    elif variant == "role_aware_verifier_weighted_mtd":
        gate_variant = "full"
    else:
        raise ValueError(f"Unknown V6 supervision variant: {variant}")
    threshold = float(gate_models[gate_variant]["selected_threshold"])
    selected = _selected_by_gate(group, gate_variant, threshold)
    if variant == "verifier_hard_filter_mtd":
        return normalized([(row, 1.0, "no_role_hard_filter") for row in selected])
    if variant == "verifier_dynamic_product_mtd":
        weighted = []
        for row in selected:
            product = (
                row["verifier_compatibility"]
                * row["diagonal_consensus_fraction"]
                * row["gate_probabilities"]["no_role"]
            )
            weighted.append((row, max(float(product), minimum_weight), "dynamic_product"))
        return normalized(weighted)
    source = "no_role_calibrated" if gate_variant == "no_role" else "role_aware_calibrated"
    return normalized(
        [
            (
                row,
                max(row["gate_probabilities"][gate_variant], minimum_weight),
                source,
            )
            for row in selected
        ]
    )


def stable_rationale_candidate(
    candidates: list[tuple[dict, float, str]], decision: str
) -> dict | None:
    compatible = [
        (row, weight)
        for row, weight, _ in candidates
        if is_valid_teacher_output(row["output"]) and row["decision"] == decision
    ]
    if not compatible:
        return None
    return max(compatible, key=lambda item: (item[1], item[0]["request_id"]))[0]


def direct_sft_answer(decision: str) -> str:
    return {
        "Supported": "The supplied evidence supports the claim.",
        "Refuted": "The supplied evidence refutes the claim.",
        "NotEnoughInfo": "The supplied evidence is insufficient to verify the claim.",
        "Answer": "The supplied evidence is sufficient to answer the recommendation query.",
        "Abstain": "The supplied evidence is insufficient for a reliable recommendation.",
    }[decision]


def assistant_content(
    decision: str,
    candidate: dict | None,
    *,
    variant: str,
    input_row: dict,
    fallback_rationale: str,
) -> str:
    answer_only = variant in {"sft_only", "answer_only_kd"}
    if variant == "sft_only":
        answer = direct_sft_answer(decision)
        rationale = ""
        evidence_id = str(input_row.get("evidence_id", "")).strip()
        citations = [evidence_id] if evidence_id and decision in {"Supported", "Refuted", "Answer"} else []
    elif candidate is None:
        answer = ""
        rationale = fallback_rationale
        citations = []
    else:
        teacher = candidate["output"]["output"]
        answer = str(teacher.get("answer", ""))
        rationale = str(teacher.get("rationale", ""))
        citations = [str(value) for value in teacher.get("cited_evidence_ids", [])]
    parts = [f"Decision: {decision}", f"Answer: {answer}"]
    if not answer_only:
        parts.append(f"Rationale: {rationale}")
    parts.append("Citations: " + (", ".join(citations) if citations else "None"))
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--config", type=Path, default=Path("configs/vgmt_v6.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = resolve(root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    student = config["student"]
    scoring = config["teacher_scoring"]

    requests_path = resolve(root, data["adjudicated_requests"])
    outputs_path = resolve(root, data["diagonal_outputs"])
    scores_path = resolve(root, data["verifier_scores"])
    gate_path = resolve(root, data["reused_gate_model"])
    oof_path = resolve(root, data["reused_gate_oof"])
    output_root = resolve(root, data["supervision_root"])
    report_path = resolve(root, data["supervision_report"])
    manifest_path = resolve(root, data["supervision_manifest"])

    request_rows = read_jsonl(requests_path)
    output_rows = read_jsonl(outputs_path)
    if len({row["request_id"] for row in request_rows}) != len(request_rows):
        raise ValueError("Duplicate request IDs in adjudicated request data")
    if len({row["request_id"] for row in output_rows}) != len(output_rows):
        raise ValueError("Duplicate request IDs in teacher output data")
    requests = {row["request_id"]: row for row in request_rows}
    outputs = {row["request_id"]: row for row in output_rows}
    if set(requests) != set(outputs):
        raise ValueError("Teacher requests and outputs do not align")

    scores = {row["source_example_id"]: row for row in read_jsonl(scores_path)}
    gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
    gate_models = gate_payload["variants"]
    for required in (scoring["primary_gate_variant"], scoring["role_ablation_gate_variant"]):
        if required not in gate_models:
            raise ValueError(f"Reused gate lacks required variant: {required}")
    if scoring["primary_gate_variant"] != "no_role":
        raise ValueError("V6 primary teacher gate must exclude role features")
    if scoring["role_ablation_gate_variant"] != "full":
        raise ValueError("V6 role ablation must use the full role-aware gate")
    oof_rows = {row["request_id"]: row for row in read_jsonl(oof_path)}

    models = tuple(config.get("models", ()))
    if not models:
        model_features = [
            name.removeprefix("model=")
            for name in gate_models["full"]["feature_names"]
            if name.startswith("model=")
        ]
        models = tuple(model_features)
    if not models:
        raise ValueError("Cannot recover frozen teacher model names from gate")

    raw_groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for request_id, output in outputs.items():
        request = requests[request_id]
        metadata = request["metadata"]
        if metadata.get("data_split") != "train":
            raise ValueError("A non-train request entered V6 supervision")
        source_id = metadata.get("source_example_id")
        if not source_id:
            raise ValueError("Request lacks source_example_id")
        raw_groups[source_id].append((request, output))
    expected = int(student["underlying_examples_per_epoch"])
    if len(raw_groups) != expected:
        raise ValueError(f"Expected {expected} V6 source groups, got {len(raw_groups)}")

    prepared: dict[str, dict] = {}
    for source_id, pairs in raw_groups.items():
        roles = {output["teacher_role"] for _, output in pairs}
        if len(pairs) != len(ROLES) or roles != set(ROLES):
            raise ValueError(f"Unbalanced diagonal teacher group: {source_id}")
        if source_id not in scores:
            raise ValueError(f"Missing verifier score for {source_id}")
        labels = {request["metadata"].get("reference_label") for request, _ in pairs}
        tasks = {request["metadata"].get("task_type") for request, _ in pairs}
        if len(labels) != 1 or len(tasks) != 1:
            raise ValueError(f"Inconsistent source supervision identity: {source_id}")
        reference_label = next(iter(labels))
        task_type = next(iter(tasks))
        decisions = Counter(output["output"]["decision"] for _, output in pairs)
        group = []
        for request, output in sorted(pairs, key=lambda item: ROLES.index(item[1]["teacher_role"])):
            decision = output["output"]["decision"]
            record = {
                "request_id": output["request_id"],
                "source_example_id": source_id,
                "task_type": task_type,
                "teacher_role": output["teacher_role"],
                "model_name": output["provenance"]["model_name"],
                "decision": decision,
                "diagonal_consensus_fraction": decisions[decision] / len(pairs),
                "output": output,
            }
            features = feature_values(record, scores[source_id], models)
            record["verifier_compatibility"] = verifier_compatibility(
                scores[source_id], task_type, decision
            )
            record["gate_probabilities"] = {}
            record["gate_probability_sources"] = {}
            for gate_variant in ("no_role", "full"):
                oof = oof_rows.get(output["request_id"], {}).get("probabilities", {}).get(gate_variant)
                if oof is None:
                    probability = predict_gate(features, gate_models[gate_variant])
                    source = "frozen_final_gate"
                else:
                    probability = float(oof)
                    source = "grouped_train_oof"
                record["gate_probabilities"][gate_variant] = probability
                record["gate_probability_sources"][gate_variant] = source
            group.append(record)
        prepared[source_id] = {
            "group": group,
            "reference_label": reference_label,
            "task_type": task_type,
            "input": group[0]["output"]["input"],
        }

    configured_variants = tuple(student["variants"])
    if configured_variants != FORMAL_VARIANTS:
        raise ValueError("V6 formal variant order differs from the method contract")
    output_files: list[Path] = []
    report_variants = {}
    minimum_weight = float(scoring["minimum_weight"])
    materialization = student["target_materialization"]
    for variant in configured_variants:
        materialized_rows = []
        decision_counts = Counter()
        selected_counts = Counter()
        selected_role_mass = Counter()
        rationale_fallbacks = 0
        for source_id in sorted(prepared):
            item = prepared[source_id]
            candidates = candidate_set(variant, item["group"], gate_models, minimum_weight)
            selected_counts[len(candidates)] += 1
            if variant == "sft_only":
                mass = Counter({item["reference_label"]: 1.0})
            else:
                mass = Counter()
                for candidate, weight, _ in candidates:
                    if is_valid_teacher_output(candidate["output"]):
                        mass[candidate["decision"]] += weight
                        selected_role_mass[candidate["teacher_role"]] += weight
                if not mass:
                    safe = "NotEnoughInfo" if item["task_type"] == "claim_verification" else "Abstain"
                    mass[safe] = 1.0
            target = hierarchical_targets_from_mass(item["task_type"], mass)
            decision = materialized_decision(target, materialization)
            decision_counts[decision] += 1
            rationale_candidate = stable_rationale_candidate(candidates, decision)
            if variant not in {"sft_only", "answer_only_kd"} and rationale_candidate is None:
                rationale_fallbacks += 1
            candidate_meta = [
                {
                    "request_id": candidate["request_id"],
                    "teacher_role": candidate["teacher_role"],
                    "model_name": candidate["model_name"],
                    "weight": weight,
                    "weight_source": source,
                    "verifier_compatibility": candidate["verifier_compatibility"],
                    "consensus_fraction": candidate["diagonal_consensus_fraction"],
                    "calibrated_correctness_probability": candidate["gate_probabilities"][
                        "full" if source == "role_aware_calibrated" else "no_role"
                    ],
                    "gate_probability_source": candidate["gate_probability_sources"][
                        "full" if source == "role_aware_calibrated" else "no_role"
                    ],
                }
                for candidate, weight, source in candidates
            ]
            for epoch in range(int(student["virtual_epochs"])):
                materialized_rows.append(
                    {
                        "conversations": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content(item["group"][0]["output"])},
                            {
                                "role": "assistant",
                                "content": assistant_content(
                                    decision,
                                    rationale_candidate,
                                    variant=variant,
                                    input_row=item["input"],
                                    fallback_rationale=materialization["safe_fallback_rationale"],
                                ),
                            },
                        ],
                        "metadata": {
                            "contract_id": TARGET_CONTRACT_ID,
                            "variant": variant,
                            "source_example_id": source_id,
                            "data_split": "train",
                            "virtual_epoch": epoch,
                            "task_type": target["task_type"],
                            "p_sufficiency": target["p_sufficiency"],
                            "p_support_given_sufficient": target["p_support_given_sufficient"],
                            "polarity_loss_weight": target["polarity_loss_weight"],
                            "p_answerable": target["p_answerable"],
                            "decision_target_distribution": target["decision_distribution"],
                            "decision_materialized_label": decision,
                            "rationale_fallback": (
                                variant not in {"sft_only", "answer_only_kd"}
                                and rationale_candidate is None
                            ),
                            "primary_gate_excludes_role": variant != "role_aware_verifier_weighted_mtd",
                            "candidates": candidate_meta,
                            "test_labels_loaded": False,
                            "legacy_dev_loaded": False,
                        },
                    }
                )
        expected_rows = expected * int(student["virtual_epochs"])
        if len(materialized_rows) != expected_rows:
            raise AssertionError(f"{variant} has {len(materialized_rows)} rows; expected {expected_rows}")
        output_path = output_root / variant / "train.jsonl"
        write_jsonl(output_path, materialized_rows)
        output_files.append(output_path)
        report_variants[variant] = {
            "source_examples": expected,
            "materialized_rows": len(materialized_rows),
            "decision_counts_per_epoch": dict(sorted(decision_counts.items())),
            "selected_candidate_count": {
                str(key): value for key, value in sorted(selected_counts.items())
            },
            "selected_role_mass": dict(sorted(selected_role_mass.items())),
            "rationale_fallbacks_per_epoch": rationale_fallbacks,
            "primary_gate_excludes_role": variant != "role_aware_verifier_weighted_mtd",
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }

    report = {
        "status": "complete",
        "contract": TARGET_CONTRACT_ID,
        "config_id": config["config_id"],
        "research_core": "verifier_guided_multi_teacher_distillation",
        "primary_method": config["evaluation"]["primary_method"],
        "primary_baseline": config["evaluation"]["primary_baseline"],
        "role_is_secondary_ablation": True,
        "examples_per_epoch": expected,
        "virtual_epochs": int(student["virtual_epochs"]),
        "formal_runs": len(configured_variants) * 3,
        "variants": report_variants,
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    def portable(path: Path) -> str:
        """Store repository-relative POSIX paths so the freeze can move hosts."""
        return path.resolve().relative_to(root).as_posix()

    manifest = {
        "config_id": config["config_id"],
        "inputs": {
            portable(path): sha256_file(path)
            for path in (config_path, requests_path, outputs_path, scores_path, gate_path, oof_path)
        },
        "outputs": {
            portable(path): sha256_file(path) for path in output_files + [report_path]
        },
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
