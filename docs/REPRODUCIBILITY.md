# Reproducibility and study versions

## Included evidence

The repository also includes 35 unchanged source/configuration files for the formal V6 pipeline and its local import dependencies. [Implementation map](IMPLEMENTATION.md) describes the reading order and external requirements. `results/code_release_manifest.json` records the SHA-256 of each published source/configuration file.

`results/decision_results.csv` exports automatic decision metrics from the frozen V6 eight-arm table. `results/human_audit_summary.csv` is a separate human audit. `results/terminal_checks.json` describes a later follow-up decision gate, not the formal 63.9M V6 result.

To inspect the headline contrasts, subtract the `unfiltered_mtd` row from `verifier_weighted_mtd`. Do not combine V6 effect sizes with the 104M development-stage model size.

## Required for independent output-level verification

- Frozen per-example predictions with seed, source/group identifier, label, model arm, and decoding configuration.
- The exact metric implementation and denominator for conditional unsafe actions.
- Annotation protocol and adjudication provenance, with the train/development/test allocation.
- Candidate identifiers, filtered-pool membership, targets, and weights for the mechanism comparison.

These row-level materials are not included in this repository. The source manifest identifies the aggregate artifacts used by the local audit.

## Required for training reproduction

Use the frozen V6 study specification and run matrix. Provide code commit, environment, data acquisition/preparation, model configuration, supervision generation, seeds, optimization budget, decoder, and evaluation commands before calling a public package training-reproducible.

Do not use the old project-root README's "24 runs pending" status as the final state: formal result tables and the completion audit describe completed experiments. Do not rewrite historical study versions into a single narrative of success.

## Interpretation limits

- 882 adjudicated task examples are different from the 314-response human rationale audit.
- Weighted versus unfiltered combines candidate filtering and weighting.
- The hard/weighted comparison is not a pure single-variable causal test: some targets/conversations differ.
- The small positive-output human comparison lacks established common-source pairing.
- Passing terminal rule/safety components does not override failed primary/mechanism checks.

## Missing portfolio metadata

Add the faculty advisor's name/department and a clear division of research responsibilities when confirmed. Faculty advice is compatible with a sole-author preprint. Preserve the status as a preprint unless its publication status changes.
