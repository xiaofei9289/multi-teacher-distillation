# VG-MTD V6 Blinded Human Grounding Audit Instructions

## Scope

This worksheet evaluates whether a model output is licensed by the evidence
shown in that row. Use only the supplied query or claim, evidence, allowed
evidence IDs, model decision, answer, rationale, and citations. Do not use
outside knowledge or search. Do not attempt to infer the system identity.

The availability-amended worksheet contains 314 rows. It preserves 20 positive
outputs per arm and contains all available safe controls up to 20 per arm. The
amendment and its supportive-only interpretation are documented separately.

## Blinding and independence

1. Annotator 1 and Annotator 2 receive separate, identical copies of
   `blind_grounding_audit.csv` and work independently.
2. Annotators must not access `hidden_key.jsonl`, system names, gold labels,
   automatic scores, or each other's completed worksheet.
3. Each annotator uses one stable anonymous `annotator_id` for every row.
4. A distinct third adjudicator receives the two completed worksheets only
   after both independent passes finish. The adjudicator may see their judgments
   but must remain blinded to system identity, gold labels, and automatic scores.
5. The adjudicator preserves agreed judgments and resolves only disagreements.
   The adjudication CSV must nevertheless contain valid entries in every
   judgment field because the deterministic scorer validates the entire sheet.
6. Do not add, delete, reorder, or edit any non-judgment cell.

## Allowed judgments

Every judgment cell must contain exactly `Yes`, `No`, or `Unclear`.

- Use `Yes` only when the supplied evidence is sufficient for the stated
  criterion.
- Use `No` when the criterion clearly fails.
- Use `Unclear` when the evidence or output is genuinely ambiguous after careful
  review. Do not use it merely to save time.

`Unclear` is scored conservatively: it fails answer support, rationale
faithfulness, and citation validity, and it counts as an unsupported material
claim for hallucination.

## Judgment fields

### `answer_supported`

Judge whether the material content of `model_answer`, together with the emitted
`model_decision`, is licensed by the supplied evidence.

- `Yes`: all material answer claims follow from the evidence and the decision is
  compatible with it.
- `No`: at least one material answer claim contradicts or exceeds the evidence,
  or the decision is incompatible with the evidence.
- `Unclear`: the evidence is too ambiguous to decide reliably.

For `NotEnoughInfo` or `Abstain`, judge whether declining a positive answer is a
defensible reading of the supplied evidence. Do not assume omitted facts.

### `rationale_faithful`

Judge whether every material proposition in `model_rationale` is supported by
the supplied evidence and whether the rationale is consistent with the emitted
decision and answer.

- `Yes`: the rationale is evidence-supported and decision-consistent.
- `No`: it contains a contradiction, unsupported material step, or a reason that
  does not support the emitted decision.
- `Unclear`: the relation cannot be decided reliably from the evidence.

A missing rationale for a positive output is `No`. A concise safe rationale that
only states that the evidence is insufficient may be `Yes` when that reading is
defensible.

### `unsupported_material_claim`

Judge whether either the answer or rationale contains at least one material
claim not licensed by the supplied evidence.

- `Yes`: at least one unsupported or contradicted material claim is present.
- `No`: every material claim is supported, or a safe output makes no material
  factual claim beyond a defensible insufficiency statement.
- `Unclear`: support cannot be decided reliably.

Minor style, grammar, or harmless wording differences are not material claims.

### `citation_valid`

Judge both identifier validity and claim-to-evidence fit.

- `Yes`: every emitted citation appears in `allowed_evidence_ids`, and the cited
  evidence supports the associated material claim.
- `No`: any citation identifier is not allowed, any cited passage fails to
  support its associated claim, or a positive material claim lacks a citation.
- `Unclear`: the citation-to-claim relation is genuinely ambiguous.

A safe output with no material factual claim and no citation may be `Yes`. An
empty citation field is `No` when the answer or rationale makes a positive
material claim.

## Completion and quality control

Before returning a worksheet, verify that:

- all 314 `audit_id` values are present exactly once;
- the column order and every non-judgment cell are unchanged;
- all four judgment fields are filled with an allowed value;
- one stable non-empty `annotator_id` is present on every row;
- notes contain no system-identity guesses or external information.

The scorer reports raw agreement and Cohen's kappa. It computes grounding
metrics only over the 20 positive outputs per arm and uses 10,000 bootstrap
replicates for the sole primary comparison,
`verifier_weighted_mtd - unfiltered_mtd`. The role-aware contrast remains
secondary and cannot rescue a failed primary claim.
