#!/usr/bin/env python3
"""Train one arm/seed under the frozen common VG-MTD V5 MiniMind recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.hierarchical_head_v5 import MiniMindWithHierarchicalHeads  # noqa: E402
from scripts.student_v5.loss_contract_v5 import LOSS_CONTRACT_ID, compute_v5_joint_loss  # noqa: E402
from scripts.student_v5.preflight_supervision_v5 import validate_preflight_report  # noqa: E402
from scripts.student_v5.vgmt_dataset_v5 import VGMTHierarchicalDatasetV5  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--base-weight-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/student_training_v5.json")
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=ROOT / "reports/vgmt_v5_supervision_preflight.json",
    )
    parser.add_argument("--log-interval", type=int, default=10)
    return parser.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(step: int, total_steps: int, base_lr: float) -> float:
    return base_lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * step / total_steps)))


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def resolve_minimind_layout(
    minimind_root: Path,
    student_config: dict,
    *,
    base_weight_override: Path | None = None,
    tokenizer_override: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve the frozen bundle-root layout without duplicating ``minimind``.

    Recipe paths are relative to the portable bundle root and already begin
    with ``minimind/``.  Python imports, however, must start inside that
    subdirectory so that ``model.model_minimind`` is importable.
    """

    code_root = minimind_root / "minimind"
    base_weight = base_weight_override or minimind_root / student_config["base_weight"]
    tokenizer_path = tokenizer_override or minimind_root / student_config["tokenizer"]
    return code_root.resolve(), base_weight.resolve(), tokenizer_path.resolve()


def main() -> None:
    args = parse_args()
    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    allowed_statuses = {
        "prospective_frozen_before_v5_training",
        "prospective_frozen_before_v6_training",
    }
    if recipe.get("status") not in allowed_statuses:
        raise ValueError("Student recipe is not in a compatible frozen prospective state")
    if args.variant not in recipe["variants"]:
        raise ValueError(f"Unknown formal student arm: {args.variant}")
    if args.seed not in recipe["optimization"]["seeds"]:
        raise ValueError(f"Seed {args.seed} is outside the frozen student seed set")
    if any(fragment in args.data_path.resolve().as_posix().casefold() for fragment in ("human_gold_dev", "human_gold_test", "test_v2")):
        raise RuntimeError("Formal trainer refuses legacy-dev and sealed-test paths")
    if args.data_path.parent.name != args.variant:
        raise ValueError("Variant id and supervision directory do not agree")

    minimind_root = args.minimind_root.resolve()
    minimind_code_root, base_weight, tokenizer_path = resolve_minimind_layout(
        minimind_root,
        recipe["student"],
        base_weight_override=args.base_weight_path,
        tokenizer_override=args.tokenizer_path,
    )
    supervision_manifest_path = ROOT / recipe["data"].get(
        "supervision_manifest", "manifests/vgmt_supervision_v5_sha256.json"
    )
    validate_preflight_report(
        args.preflight_report.resolve(),
        recipe_path=recipe_path,
        supervision_manifest_path=supervision_manifest_path,
        tokenizer_path=tokenizer_path,
    )
    if not (minimind_code_root / "model/model_minimind.py").exists():
        raise FileNotFoundError(
            f"MiniMind bundle root must contain minimind/model/model_minimind.py: {minimind_root}"
        )
    if sha256(base_weight) != recipe["student"]["base_weight_sha256"]:
        raise RuntimeError("MiniMind base checkpoint hash differs from the frozen recipe")
    sys.path.insert(0, str(minimind_code_root))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for V5 formal training")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    setup_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    optimization = recipe["optimization"]
    loss_config = recipe["loss"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model = MiniMindForCausalLM(
        MiniMindConfig(
            hidden_size=int(recipe["student"]["hidden_size"]),
            num_hidden_layers=int(recipe["student"]["num_hidden_layers"]),
            use_moe=bool(recipe["student"]["use_moe"]),
        )
    )
    base_model.load_state_dict(torch.load(base_weight, map_location="cpu"), strict=True)
    model = MiniMindWithHierarchicalHeads(base_model, int(recipe["student"]["hidden_size"])).to(device)
    base_parameter_count = sum(parameter.numel() for parameter in model.base_model.parameters())
    head_parameter_count = sum(parameter.numel() for parameter in model.decision_heads.parameters())
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_counts = (
        int(recipe["student"]["base_parameter_count"]),
        int(recipe["student"]["hierarchical_head_parameter_count"]),
        int(recipe["student"]["total_parameter_count"]),
    )
    if (base_parameter_count, head_parameter_count, total_parameter_count) != expected_counts:
        raise RuntimeError("MiniMind V5 parameter counts differ from the frozen recipe")
    dataset = VGMTHierarchicalDatasetV5(
        args.data_path,
        tokenizer,
        max_length=int(optimization["max_seq_len"]),
        max_prompt_tokens=int(optimization["max_prompt_tokens"]),
    )
    if len(dataset) != int(recipe["data"]["rows_per_variant"]):
        raise RuntimeError(f"Expected {recipe['data']['rows_per_variant']} training rows, got {len(dataset)}")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(optimization["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(optimization["num_workers"]),
        pin_memory=True,
        persistent_workers=int(optimization["num_workers"]) > 0,
    )
    optimizer = optim.AdamW(
        [
            {"params": model.base_model.parameters(), "lr": float(optimization["base_learning_rate"]), "group_name": "base"},
            {"params": model.decision_heads.parameters(), "lr": float(optimization["head_learning_rate"]), "group_name": "hierarchical_heads"},
        ],
        weight_decay=float(optimization["weight_decay"]),
    )
    dtype = optimization["dtype"]
    if dtype not in {"float16", "bfloat16"}:
        raise ValueError("Formal training dtype must be float16 or bfloat16")
    use_fp16 = dtype == "float16"
    if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen bfloat16 recipe requires a CUDA device with bfloat16 support")
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    accumulation = int(optimization["accumulation_steps"])
    total_updates = int(optimization["epochs"]) * math.ceil(len(loader) / accumulation)
    update_step = 0
    started = time.time()
    running = CounterLike()
    totals = CounterLike()
    log_path = args.run_dir / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)

    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle:
        for epoch in range(int(optimization["epochs"])):
            model.train()
            remainder = len(loader) % accumulation
            final_window_start = len(loader) - remainder + 1 if remainder else len(loader) + 1
            for micro_step, batch in enumerate(loader, 1):
                (
                    input_ids,
                    labels,
                    decision_mask,
                    task_ids,
                    sufficiency_targets,
                    polarity_targets,
                    polarity_weights,
                    answerability_targets,
                    attention_mask,
                ) = batch
                tensors = [input_ids, labels, decision_mask, task_ids, sufficiency_targets, polarity_targets, polarity_weights, answerability_targets]
                tensors = [tensor.to(device, non_blocking=True) for tensor in tensors]
                (
                    input_ids,
                    labels,
                    decision_mask,
                    task_ids,
                    sufficiency_targets,
                    polarity_targets,
                    polarity_weights,
                    answerability_targets,
                ) = tensors
                del attention_mask
                divisor = remainder if remainder and micro_step >= final_window_start else accumulation
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    output = model(input_ids, task_ids, labels=labels, use_cache=False)
                    joint = compute_v5_joint_loss(
                        output.logits,
                        labels,
                        decision_mask,
                        output.decisions,
                        sufficiency_targets,
                        polarity_targets,
                        polarity_weights,
                        answerability_targets,
                        generation_loss_weight=float(loss_config["generation_weight"]),
                        sufficiency_loss_weight=float(loss_config["sufficiency_weight"]),
                        polarity_loss_weight=float(loss_config["polarity_weight"]),
                        answerability_loss_weight=float(loss_config["answerability_weight"]),
                        sufficiency_positive_weight=float(loss_config["common_positive_weights_from_human_train_priors"]["sufficiency"]),
                        polarity_positive_weight=float(loss_config["common_positive_weights_from_human_train_priors"]["polarity_support"]),
                        answerability_positive_weight=float(loss_config["common_positive_weights_from_human_train_priors"]["answerability"]),
                        aux_loss=output.aux_loss,
                    )
                    raw_loss = joint.total_loss
                if not torch.isfinite(raw_loss):
                    raise RuntimeError(f"Non-finite V5 loss at epoch={epoch} micro_step={micro_step}")
                scaler.scale(raw_loss / divisor).backward()
                running.add("loss", raw_loss.item())
                running.add("generation_loss", joint.generation_loss.item())
                running.add("decision_loss", joint.decision_loss.item())
                running.add("batches", 1)
                totals.add("generation_tokens", joint.generation_valid_token_count)
                totals.add("excluded_decision_tokens", joint.excluded_decision_token_count)
                totals.add("claim_rows", joint.decision_components.claim_count)
                totals.add("query_rows", joint.decision_components.query_count)
                should_step = micro_step % accumulation == 0 or micro_step == len(loader)
                if not should_step:
                    continue
                update_step += 1
                base_lr = cosine_lr(update_step, total_updates, float(optimization["base_learning_rate"]))
                head_lr = cosine_lr(update_step, total_updates, float(optimization["head_learning_rate"]))
                optimizer.param_groups[0]["lr"] = base_lr
                optimizer.param_groups[1]["lr"] = head_lr
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["grad_clip"]), error_if_nonfinite=True)
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) < previous_scale:
                    raise RuntimeError(f"GradScaler skipped formal update {update_step}")
                optimizer.zero_grad(set_to_none=True)
                batches = running["batches"]
                event = {
                    "variant": args.variant,
                    "seed": args.seed,
                    "epoch": epoch,
                    "update_step": update_step,
                    "total_updates": total_updates,
                    "loss": running["loss"] / batches,
                    "generation_loss": running["generation_loss"] / batches,
                    "decision_loss": running["decision_loss"] / batches,
                    "base_learning_rate": base_lr,
                    "head_learning_rate": head_lr,
                    "grad_norm": float(grad_norm),
                    "loss_contract_id": LOSS_CONTRACT_ID,
                    "elapsed_seconds": time.time() - started,
                    "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                }
                log_handle.write(json.dumps(event, sort_keys=True) + "\n")
                log_handle.flush()
                if update_step == 1 or update_step % args.log_interval == 0 or update_step == total_updates:
                    print(f"{args.variant} seed={args.seed} update {update_step}/{total_updates} loss={event['loss']:.4f}", flush=True)
                totals.add("update_loss", event["loss"])
                totals.add("update_generation_loss", event["generation_loss"])
                totals.add("update_decision_loss", event["decision_loss"])
                running.clear()

    model.eval()
    weight_path = args.run_dir / "model.pth"
    atomic_torch_save({key: value.detach().half().cpu() for key, value in model.state_dict().items()}, weight_path)
    summary = {
        "status": "completed",
        "recipe_id": recipe["config_id"],
        "loss_contract_id": LOSS_CONTRACT_ID,
        "variant": args.variant,
        "seed": args.seed,
        "data_path": str(args.data_path.resolve()),
        "data_sha256": sha256(args.data_path),
        "data_rows": len(dataset),
        "base_weight_sha256": sha256(base_weight),
        "base_parameter_count": base_parameter_count,
        "hierarchical_head_parameter_count": head_parameter_count,
        "total_parameter_count": total_parameter_count,
        "output_weight": str(weight_path.resolve()),
        "output_weight_sha256": sha256(weight_path),
        "optimizer_updates": update_step,
        "mean_update_loss": totals["update_loss"] / update_step,
        "mean_generation_loss": totals["update_generation_loss"] / update_step,
        "mean_decision_loss": totals["update_decision_loss"] / update_step,
        "generation_tokens_seen": totals["generation_tokens"],
        "excluded_decision_tokens_seen": totals["excluded_decision_tokens"],
        "claim_rows_seen": totals["claim_rows"],
        "query_rows_seen": totals["query_rows"],
        "elapsed_seconds": time.time() - started,
        "gpu_name": torch.cuda.get_device_name(0),
        "dtype": dtype,
        "test_labels_loaded": False,
        "legacy_dev_loaded": False,
        "supervision_preflight": {
            "path": str(args.preflight_report.resolve()),
            "sha256": sha256(args.preflight_report.resolve()),
        },
        "files": {
            "recipe_sha256": sha256(recipe_path),
            "trainer_sha256": sha256(Path(__file__)),
            "dataset_sha256": sha256(ROOT / "scripts/student_v5/vgmt_dataset_v5.py"),
            "inference_io_sha256": sha256(ROOT / "scripts/student_v5/inference_io_v5.py"),
            "head_sha256": sha256(ROOT / "scripts/student_v5/hierarchical_head_v5.py"),
            "loss_sha256": sha256(ROOT / "scripts/student_v5/loss_contract_v5.py"),
        },
    }
    (args.run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


class CounterLike(dict):
    def __missing__(self, key):
        return 0.0

    def add(self, key, value):
        self[key] = self[key] + value


if __name__ == "__main__":
    main()
