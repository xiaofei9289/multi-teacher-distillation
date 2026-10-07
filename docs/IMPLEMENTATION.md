# Implementation map

The formal V6 pipeline reuses the V5 hierarchical student components and earlier decision-contract utilities. Version numbers in module names describe code lineage; the experiment recipe in this repository is the formal V6 recipe. The 35 source/configuration files were copied unchanged, with hashes in [code_release_manifest.json](../results/code_release_manifest.json).

## Suggested reading order

| Stage | Entry point | What to inspect |
|---|---|---|
| Supervision | [build_verifier_core_supervision_v6.py](../scripts/student_v6/build_verifier_core_supervision_v6.py) | Eight arms, candidate validity, correctness gate, targets, and weighting |
| Data/loss | [vgmt_dataset_v5.py](../scripts/student_v5/vgmt_dataset_v5.py), [loss_contract_v5.py](../scripts/student_v5/loss_contract_v5.py) | Token masks, rationale supervision, and hierarchical decision losses |
| Student training | [train_hierarchical_student_v5.py](../scripts/student_v5/train_hierarchical_student_v5.py) | Recipe validation, optimization, numerical guards, and run provenance |
| Experiment matrix | [run_formal_matrix_v6.py](../scripts/student_v6/run_formal_matrix_v6.py) | Eight arms × three seeds, preflight checks, and checkpoint validation |
| Evaluation freeze | [freeze_sealed_evaluation_v6.py](../scripts/student_v6/freeze_sealed_evaluation_v6.py) | Frozen checkpoint identities before sealed evaluation |
| Automatic evaluation | [evaluate_grounded_predictions_v6.py](../scripts/student_v6/evaluate_grounded_predictions_v6.py) | Decision metrics, label recall, and task-defined conditional risk |
| Human audit | [score_blinded_grounding_audit_v6_availability_amendment.py](../scripts/student_v6/score_blinded_grounding_audit_v6_availability_amendment.py) | Positive-output/safe-abstention audit and availability amendment |

## Execution prerequisites

The original run used an external MiniMind checkout/model/tokenizer, PyTorch, Transformers, and additional scoring dependencies visible in the source. Exact package versions and the external code revision have not been exported as a validated environment lock here. Do not interpret this snapshot as a one-command training release.

The launcher additionally requires the original generated V6 supervision JSONL files, their SHA-256 manifest, and a matching preflight report. Evaluation requires the frozen checkpoint manifest and sealed examples. These inputs, raw annotations, and checkpoints are not included. Keep train/development/test provenance separate when preparing them.

The repository's supported lightweight check is `python scripts/verify_results.py`, which verifies the exported aggregate results using the Python standard library. Publication preparation checked the syntax of the exported Python modules and JSON configuration; training was not rerun.
