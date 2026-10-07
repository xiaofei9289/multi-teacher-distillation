#!/usr/bin/env python3
"""Score the availability-amended V6 blind audit without confirmatory promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file  # noqa: E402
from scripts.student_v6.build_blinded_grounding_audit_v6 import (  # noqa: E402
    BLIND_COLUMNS,
    POSITIVE,
    SAFE,
)
from scripts.student_v6.build_blinded_grounding_audit_v6_availability_amendment import (  # noqa: E402
    DEFAULT_CONFIG,
    FROZEN_BUILDER,
    FROZEN_CONFIG,
    FROZEN_SCORER,
    validate_amendment_contract,
)
from scripts.student_v6.score_blinded_grounding_audit_v6 import (  # noqa: E402
    JUDGMENT_FIELDS,
    bootstrap_difference,
    cohen_kappa,
    indexed,
    mean_metrics,
    read_csv,
    resolve_judgments,
    validate_completed_sheet,
)


def claim_state(
    required_automatic: dict[str, bool], human_gates: dict[str, bool]
) -> dict[str, object]:
    supportive = all(required_automatic.values()) and all(human_gates.values())
    return {
        "availability_amended_supportive_claim_supported": supportive,
        "confirmatory_positive_claim_authorized": False,
        "full_positive_claim_authorized": False,
        "result_state": (
            "availability-amended-supportive-evidence-supported"
            if supportive
            else "availability-amended-supportive-evidence-not-supported"
        ),
    }


def agreement_for_ids(
    audit_ids: list[str],
    first: dict[str, dict[str, str]],
    second: dict[str, dict[str, str]],
    disagreements: dict[str, int] | None = None,
) -> dict[str, dict[str, float | int]]:
    if not audit_ids:
        raise ValueError("Agreement requires at least one audit row")
    return {
        field: {
            "raw_agreement": sum(
                first[audit_id][field] == second[audit_id][field]
                for audit_id in audit_ids
            )
            / len(audit_ids),
            "cohen_kappa": cohen_kappa(
                [first[audit_id][field] for audit_id in audit_ids],
                [second[audit_id][field] for audit_id in audit_ids],
            ),
            **(
                {"disagreements": disagreements[field]}
                if disagreements is not None
                else {}
            ),
        }
        for field in JUDGMENT_FIELDS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-csv", type=Path, required=True)
    parser.add_argument("--annotator-one", type=Path, required=True)
    parser.add_argument("--annotator-two", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--hidden-key", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--automatic-results", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports/vgmt_v6_human_grounding_results_availability_amendment.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    frozen_config = json.loads(args.frozen_config.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.audit_manifest.read_text(encoding="utf-8"))
    automatic = json.loads(args.automatic_results.read_text(encoding="utf-8"))
    validate_amendment_contract(config, frozen_config)

    if freeze.get("human_audit_builder_sha256") != sha256_file(FROZEN_BUILDER):
        raise RuntimeError("Frozen human audit builder changed after freeze")
    if freeze.get("human_audit_scorer_sha256") != sha256_file(FROZEN_SCORER):
        raise RuntimeError("Frozen human audit scorer changed after freeze")
    if freeze.get("human_audit_config", {}).get("sha256") != sha256_file(
        args.frozen_config
    ):
        raise RuntimeError("Frozen human audit config changed after freeze")
    if audit.get("status") != "complete" or audit.get("contract") != config.get(
        "config_id"
    ):
        raise RuntimeError("Availability-amended audit pack is incomplete")
    if audit.get("amendment_id") != config.get("amendment_id"):
        raise RuntimeError("Audit pack and availability amendment IDs differ")
    if audit.get("builder_sha256") != sha256_file(
        ROOT
        / "scripts/student_v6/build_blinded_grounding_audit_v6_availability_amendment.py"
    ):
        raise RuntimeError("Availability-amended audit builder changed")
    if audit.get("amendment_config", {}).get("sha256") != sha256_file(args.config):
        raise RuntimeError("Audit pack used a different availability amendment config")
    if audit.get("frozen_parent", {}).get("config_sha256") != sha256_file(
        args.frozen_config
    ):
        raise RuntimeError("Audit pack used a different frozen parent config")
    if audit.get("confirmatory_positive_claim_authorization_prohibited") is not True:
        raise RuntimeError("Audit pack does not prohibit confirmatory promotion")
    if not all(
        audit.get(key) is True
        for key in (
            "system_identity_blinded",
            "gold_labels_blinded",
            "automatic_scores_blinded",
            "positive_stratum_unchanged",
            "safe_control_stratum_only_is_availability_capped",
        )
    ):
        raise RuntimeError("Audit pack did not preserve the amended blind contract")
    if audit.get("test_labels_used_for_sampling") is not False or audit.get(
        "automatic_scores_used_for_sampling"
    ) is not False:
        raise RuntimeError("Audit sampling used prohibited information")
    if audit.get("blind_csv", {}).get("sha256") != sha256_file(args.blind_csv):
        raise RuntimeError("Availability-amended blind worksheet changed")
    if audit.get("hidden_key", {}).get("sha256") != sha256_file(args.hidden_key):
        raise RuntimeError("Availability-amended hidden key changed")
    if automatic.get("status") != "complete" or automatic.get(
        "full_positive_claim_authorized"
    ) is not False:
        raise RuntimeError("Automatic V6 summary is missing or improperly authorizes a claim")
    if automatic.get("provenance", {}).get("freeze_manifest", {}).get(
        "sha256"
    ) != sha256_file(args.freeze_manifest):
        raise RuntimeError("Automatic summary and human audit use different freezes")

    columns, base_rows = read_csv(args.blind_csv)
    if tuple(columns) != BLIND_COLUMNS:
        raise ValueError("Blind worksheet columns differ from the frozen schema")
    base = indexed(base_rows, label="blind worksheet")
    if len(base) != int(audit["rows"]):
        raise RuntimeError("Blind worksheet row count differs from audit manifest")
    first, first_id = validate_completed_sheet(
        args.annotator_one, base, label="annotator one"
    )
    second, second_id = validate_completed_sheet(
        args.annotator_two, base, label="annotator two"
    )
    adjudicated, adjudicator_id = validate_completed_sheet(
        args.adjudicated, base, label="adjudication"
    )
    if len({first_id, second_id, adjudicator_id}) != 3:
        raise ValueError("Annotator and adjudicator IDs must be distinct")
    final, disagreements = resolve_judgments(first, second, adjudicated)

    hidden_rows = read_jsonl(args.hidden_key)
    hidden = {row["audit_id"]: row for row in hidden_rows}
    if len(hidden) != len(hidden_rows):
        raise ValueError("Hidden key contains duplicate audit IDs")
    if set(hidden) != set(base):
        raise ValueError("Hidden key and blind worksheet audit IDs differ")
    if {row.get("variant") for row in hidden_rows} != set(config["formal_variants"]):
        raise RuntimeError("Hidden key variants differ from the amended contract")
    for audit_id, row in hidden.items():
        stratum = row.get("sampling_stratum")
        if stratum == "positive" and row.get("prediction") not in POSITIVE:
            raise RuntimeError(f"Positive stratum contains a safe output: {audit_id}")
        if stratum == "safe" and row.get("prediction") not in SAFE:
            raise RuntimeError(f"Safe stratum contains a positive output: {audit_id}")
        if stratum not in {"positive", "safe"}:
            raise RuntimeError(f"Hidden key lacks a valid sampling stratum: {audit_id}")

    joined = [
        {**hidden[audit_id], "judgment": final[audit_id]}
        for audit_id in sorted(base)
    ]
    all_ids = sorted(base)
    positive_ids = sorted(
        audit_id
        for audit_id, row in hidden.items()
        if row["sampling_stratum"] == "positive"
    )
    agreement = agreement_for_ids(all_ids, first, second, disagreements)
    positive_agreement = agreement_for_ids(positive_ids, first, second)

    by_variant: dict[str, dict[str, float | int]] = {}
    positives_by_variant: dict[str, list[dict]] = {}
    positive_target = int(config["positive_outputs_per_variant"])
    safe_target = int(config["safe_outputs_target_per_variant"])
    strata_manifest = audit.get("strata_by_variant", {})
    expected_total = sum(
        int(counts.get("positive_selected", -1))
        + int(counts.get("safe_selected", -1))
        for counts in strata_manifest.values()
    )
    if set(strata_manifest) != set(config["formal_variants"]) or expected_total != len(
        base
    ):
        raise RuntimeError("Audit manifest strata do not account for every worksheet row")
    for variant in config["formal_variants"]:
        rows = [row for row in joined if row["variant"] == variant]
        positives = [row for row in rows if row["sampling_stratum"] == "positive"]
        safe = [row for row in rows if row["sampling_stratum"] == "safe"]
        counts = strata_manifest.get(variant, {})
        expected_safe = min(safe_target, int(counts.get("safe_available", -1)))
        if len(positives) != positive_target or len(safe) != expected_safe:
            raise ValueError(f"Human audit strata differ for {variant}")
        if counts.get("positive_selected") != len(positives) or counts.get(
            "safe_selected"
        ) != len(safe):
            raise ValueError(f"Human audit manifest counts differ for {variant}")
        positives_by_variant[variant] = positives
        by_variant[variant] = {
            "audited_rows": len(rows),
            "positive_rows": len(positives),
            "safe_rows": len(safe),
            **mean_metrics(positives),
        }

    candidate = automatic["primary_method"]
    baseline = automatic["primary_baseline"]
    comparison = bootstrap_difference(
        positives_by_variant[candidate],
        positives_by_variant[baseline],
        replicates=int(config["human_comparison_bootstrap_replicates"]),
        confidence=float(config["confidence"]),
        seed=int(config["seed"]) + 1,
    )
    human_gates = {
        "claim_support": comparison["claim_support_rate"]["ci_low"] > 0.0,
        "rationale_faithfulness": comparison["rationale_faithfulness"]["ci_low"]
        > 0.0,
        "hallucination": comparison["hallucination_rate"]["ci_high"] < 0.0,
    }
    automatic_gates = automatic["primary_automatic_gates"]
    required_automatic = {
        key: bool(automatic_gates[key])
        for key in (
            "decision_quality",
            "unsafe_answer_noninferiority",
            "abstention_f1_noninferiority",
            "all_label_recalls_nonzero",
        )
    }
    payload = {
        "status": "complete",
        "contract": "vgmtd_v6_blinded_human_grounding_summary_availability_amendment",
        "amendment_id": config["amendment_id"],
        "interpretation_scope": config["interpretation_scope"],
        "pre_registered_protocol_executed": False,
        "primary_method": candidate,
        "primary_baseline": baseline,
        "role_is_secondary_ablation": True,
        "unclear_policy": config["unclear_policy"],
        "agreement_all_amended_rows": agreement,
        "agreement_positive_outputs": positive_agreement,
        "adjudication": {
            "annotator_ids": [first_id, second_id],
            "adjudicator_id": adjudicator_id,
            "system_identity_unblinded_before_adjudication": False,
        },
        "metrics_by_variant": by_variant,
        "primary_human_comparison": comparison,
        "primary_human_gates": human_gates,
        "required_automatic_gates": required_automatic,
        "automatic_grounding_diagnostics_not_used_for_authorization": True,
        **claim_state(required_automatic, human_gates),
        "provenance": {
            "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
            "automatic_results_sha256": sha256_file(args.automatic_results),
            "audit_manifest_sha256": sha256_file(args.audit_manifest),
            "blind_csv_sha256": sha256_file(args.blind_csv),
            "annotator_one_sha256": sha256_file(args.annotator_one),
            "annotator_two_sha256": sha256_file(args.annotator_two),
            "adjudicated_sha256": sha256_file(args.adjudicated),
            "hidden_key_sha256": sha256_file(args.hidden_key),
            "frozen_config_sha256": sha256_file(args.frozen_config),
            "amendment_config_sha256": sha256_file(args.config),
            "scorer_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
