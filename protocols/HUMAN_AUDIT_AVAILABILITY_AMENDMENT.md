# VG-MTD V6 Human-Audit Availability Amendment

**Amendment ID:** `vgmt_v6_human_audit_availability_amendment_20260828`
**Recorded:** 2026-08-28, after sealed prediction and automatic summary, before
any human annotation
**Parent contract:** `vgmt_v6_blinded_grounding_audit`

## Trigger and scope

The frozen audit requested, for each of eight arms at reference training seed
260631, 20 positive outputs and 20 safe outputs. The frozen builder stopped
before writing an audit pack because `single_teacher_rationale_kd` produced only
14 safe outputs. Repeating responses, changing the reference seed, substituting
another arm, or selecting with gold labels would make the worksheet appear
balanced while weakening its provenance; none is allowed.

This amendment changes only the unavailable safe-control stratum:

- positive outputs remain exactly 20 per arm, sampled with the frozen seed and
  frozen stable ordering;
- safe outputs are sampled without replacement as `min(20, available)`;
- the sole primary comparison remains
  `verifier_weighted_mtd - unfiltered_mtd`;
- `role_aware_verifier_weighted_mtd - verifier_weighted_mtd` remains secondary;
- system identity, gold labels, and automatic support scores remain blinded;
- two independent annotators and a distinct blinded adjudicator remain required;
- `Unclear` retains the frozen conservative interpretation.

No human metric uses the safe-control rows. The primary human comparison remains
based on 20 independently sampled positive outputs for each side. The reduced
safe stratum affects worksheet size and overall agreement denominators only.

## Frozen availability observation

Counts below use model predictions only. Gold labels and automatic grounding
scores were not used for sample eligibility or ordering.

| Arm | Positive available | Safe available | Positive selected | Safe selected |
|---|---:|---:|---:|---:|
| `sft_only` | 159 | 108 | 20 | 20 |
| `answer_only_kd` | 246 | 21 | 20 | 20 |
| `single_teacher_rationale_kd` | 253 | 14 | 20 | 14 |
| `unfiltered_mtd` | 171 | 96 | 20 | 20 |
| `verifier_hard_filter_mtd` | 83 | 184 | 20 | 20 |
| `verifier_weighted_mtd` | 50 | 217 | 20 | 20 |
| `verifier_dynamic_product_mtd` | 56 | 211 | 20 | 20 |
| `role_aware_verifier_weighted_mtd` | 54 | 213 | 20 | 20 |

The amended worksheet therefore contains 314 unique responses rather than 320:
160 positive outputs and 154 safe outputs.

## Epistemic consequence

This is a post-unsealing availability amendment. Even though it preserves the
positive-output sample sizes used by the declared human comparison, it is not
the exact preregistered 320-response procedure. Accordingly:

- `full_positive_claim_authorized` is permanently `false` for this amended path;
- human results may be reported only as availability-amended supportive evidence;
- the original builder, scorer, and config remain byte-identical and continue to
  document that the exact frozen protocol was infeasible;
- the paper must disclose both the 14/20 shortfall and the 314-row amendment;
- the automatic result cannot be replaced or rescued by the secondary role arm.

The automatic summary was already available when this amendment was recorded
and had state `automatic-gates-fail-or-inconclusive`. In particular, the primary
decision-quality and safety gates passed, while all-label recall and all three
automatic grounding gates did not. This timing is disclosed so the amendment is
not presented as prospectively preregistered.
