#!/usr/bin/env python3
"""Fail closed when implementation or paper metadata drifts from V6's mainline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]

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
FORMAL_SEEDS = (260631, 260632, 260633)
PRIMARY_METHOD = "verifier_weighted_mtd"
PRIMARY_BASELINE = "unfiltered_mtd"
ROLE_METHOD = "role_aware_verifier_weighted_mtd"


class MainlineDriftError(RuntimeError):
    """Raised when a V6 artifact silently changes the frozen research question."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MainlineDriftError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path = ROOT) -> dict:
    matrix = yaml.safe_load((root / "configs/experiment_matrix.yaml").read_text(encoding="utf-8"))
    method = load_json(root / "configs/vgmt_v6.json")
    recipe = load_json(root / "configs/student_training_v6.json")
    audit = load_json(root / "configs/human_grounding_audit_v6.json")
    paper = load_json(root / "paper_rewriting_output/paper_spine_config.json")

    matrix_variants = tuple(item["id"] for item in matrix["formal_variants"])
    require(matrix.get("scope_id") == "vg_mtd_scope_v6", "experiment matrix is not V6")
    require(matrix.get("paper_core") == "verifier_guided_multi_teacher_rationale_distillation", "paper core drifted")
    require(matrix_variants == FORMAL_VARIANTS, "formal variant order or membership drifted")
    require(tuple(matrix["formal_seeds"]) == FORMAL_SEEDS, "formal seeds drifted")
    require(int(matrix["formal_run_count"]) == 24, "formal run count drifted")

    primary = [item for item in matrix["required_comparisons"] if item.get("primary")]
    require(len(primary) == 1, "V6 must have exactly one primary comparison")
    require(primary[0]["left"] == PRIMARY_METHOD and primary[0]["right"] == PRIMARY_BASELINE, "primary comparison drifted")
    require(matrix["guardrails"]["role_features_are_secondary_ablation"] is True, "role ablation was promoted")
    require(matrix["guardrails"]["role_value_is_headline_gate"] is False, "role value became a headline gate")

    require(tuple(method["student"]["variants"]) == FORMAL_VARIANTS, "method variants drifted")
    require(method["teacher_scoring"]["primary_gate_variant"] == "no_role", "primary teacher gate became role-aware")
    require(method["evaluation"]["primary_method"] == PRIMARY_METHOD, "method primary arm drifted")
    require(method["evaluation"]["primary_baseline"] == PRIMARY_BASELINE, "method primary baseline drifted")
    require(method["evaluation"]["role_comparison"] == [ROLE_METHOD, PRIMARY_METHOD], "role comparison drifted")
    require(method["evaluation"]["role_null_is_compatible_with_core_claim"] is True, "role-null interpretation drifted")

    require(tuple(recipe["variants"]) == FORMAL_VARIANTS, "training variants drifted")
    require(tuple(recipe["optimization"]["seeds"]) == FORMAL_SEEDS, "training seeds drifted")
    require(int(recipe["optimization"]["formal_run_count"]) == 24, "training run count drifted")
    require(recipe["reporting"]["primary_comparison"] == "verifier_weighted_mtd_minus_unfiltered_mtd", "reporting contrast drifted")
    require(recipe["reporting"]["role_comparison_is_secondary"] is True, "role reporting was promoted")
    require(tuple(audit["formal_variants"]) == FORMAL_VARIANTS, "human audit variants drifted")
    require(int(audit["independent_annotators"]) == 2, "human audit no longer has two independent annotators")
    require(audit["adjudicate_disagreements"] is True, "human audit adjudication was disabled")
    require(audit["system_unblinding_before_adjudication"] is False, "human audit may unblind systems before adjudication")
    require(int(audit["human_comparison_bootstrap_replicates"]) == 10000, "human audit uncertainty budget drifted")
    require((root / "scripts/student_v6/score_blinded_grounding_audit_v6.py").exists(), "human audit scorer is missing")

    motivation = paper["user_motivation"]
    require("verifier-guided" in motivation and "unfiltered" in motivation, "paper motivation lost the verifier contrast")
    require("secondary" in motivation and "role" in motivation.lower(), "paper motivation lost the role boundary")
    require(paper["draft_path"] == "docs/paper_vgmt_v6/MANUSCRIPT.md", "paper draft path is not V6")

    return {
        "status": "passed",
        "scope_id": matrix["scope_id"],
        "primary_comparison": f"{PRIMARY_METHOD}_minus_{PRIMARY_BASELINE}",
        "role_comparison": f"{ROLE_METHOD}_minus_{PRIMARY_METHOD}",
        "formal_variants": len(FORMAL_VARIANTS),
        "formal_seeds": len(FORMAL_SEEDS),
        "formal_runs": 24,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
