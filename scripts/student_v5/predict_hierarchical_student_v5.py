#!/usr/bin/env python3
"""Produce frozen V5 hierarchical-head predictions for one checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.student_v5.hierarchical_head_v5 import (  # noqa: E402
    CLAIM_LABELS,
    CLAIM_TASK_ID,
    QUERY_LABELS,
    QUERY_TASK_ID,
    MiniMindWithHierarchicalHeads,
)
from scripts.student_v5.vgmt_v5_contract import read_jsonl, sha256_file, write_jsonl  # noqa: E402
from scripts.student_v5.inference_io_v5 import (  # noqa: E402
    build_eval_conversations,
    cap_conversations,
    clean_empty_think,
    render_generation_prefix,
)


TEST_FRAGMENTS = ("human_gold_test", "test_v2", "sealed_test")


def evidence_condition(row: dict) -> tuple[str, str]:
    metadata = row.get("metadata", {})
    raw = (
        metadata.get("construction_method")
        or metadata.get("scenario")
        or metadata.get("claim_type")
        or row.get("query_type")
        or "unspecified"
    )
    lowered = str(raw).casefold()
    if "stale" in lowered or "temporal" in lowered:
        category = "stale"
    elif "contradict" in lowered or "conflict" in lowered:
        category = "conflicting"
    elif any(token in lowered for token in ("sparse", "mismatch", "out_of_ontology", "insufficient", "unsupported")):
        category = "insufficient"
    elif row.get("gold_label") in {"NotEnoughInfo", "Abstain"}:
        category = "insufficient"
    else:
        category = "ordinary"
    return str(raw), category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimind-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--recipe", type=Path, default=ROOT / "configs/student_training_v5.json")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sealed-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    return parser.parse_args()


def authorize_test(args: argparse.Namespace) -> dict | None:
    is_test = any(fragment in args.eval_data.resolve().as_posix().casefold() for fragment in TEST_FRAGMENTS)
    if not is_test and not args.sealed_test:
        return None
    if not args.sealed_test or args.freeze_manifest is None:
        raise RuntimeError("Sealed-test inference requires --sealed-test and --freeze-manifest")
    manifest = json.loads(args.freeze_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen" or manifest.get("test_open_authorized") is not True:
        raise RuntimeError("V5 freeze manifest does not authorize sealed-test inference")
    if manifest.get("predictor_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Predictor implementation changed after the sealed freeze")
    checkpoint_hash = sha256_file(args.weight)
    if checkpoint_hash not in set(manifest.get("authorized_checkpoint_sha256", [])):
        raise RuntimeError("Checkpoint is not authorized by the sealed freeze")
    return manifest


def decode(task_type: str, probabilities: dict[str, float], decoding: dict) -> str:
    minimum = float(decoding["positive_min_probability"])
    margin = float(decoding["safety_margin"])
    if task_type == "query_decision":
        return "Answer" if probabilities["Answer"] >= minimum and probabilities["Answer"] - probabilities["Abstain"] >= margin else "Abstain"
    positive = max(("Supported", "Refuted"), key=lambda label: (probabilities[label], -CLAIM_LABELS.index(label)))
    return positive if probabilities[positive] >= minimum and probabilities[positive] - probabilities["NotEnoughInfo"] >= margin else "NotEnoughInfo"


def main() -> None:
    args = parse_args()
    freeze = authorize_test(args)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    minimind_root = args.minimind_root.resolve()
    bundled_code_root = minimind_root / "minimind"
    if (bundled_code_root / "model/model_minimind.py").exists():
        code_root = bundled_code_root
        default_tokenizer = minimind_root / recipe["student"]["tokenizer"]
    elif (minimind_root / "model/model_minimind.py").exists():
        code_root = minimind_root
        default_tokenizer = minimind_root / "model"
    else:
        raise FileNotFoundError(
            "MiniMind root must be either a portable bundle containing minimind/ "
            "or the minimind code directory itself"
        )
    tokenizer_path = (args.tokenizer_path or default_tokenizer).resolve()
    sys.path.insert(0, str(code_root))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for V5 checkpoint inference")
    rows = read_jsonl(args.eval_data)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    base_model = MiniMindForCausalLM(
        MiniMindConfig(
            hidden_size=int(recipe["student"]["hidden_size"]),
            num_hidden_layers=int(recipe["student"]["num_hidden_layers"]),
            use_moe=bool(recipe["student"]["use_moe"]),
        )
    )
    model = MiniMindWithHierarchicalHeads(base_model, int(recipe["student"]["hidden_size"]))
    model.load_state_dict(torch.load(args.weight, map_location="cpu"), strict=True)
    model = model.half().eval().to(device)
    predictions = []
    for index, row in enumerate(rows, 1):
        raw_condition, reliability_condition = evidence_condition(row)
        conversations, cap_info = cap_conversations(
            tokenizer,
            build_eval_conversations(row),
            int(recipe["optimization"]["max_prompt_tokens"]),
        )
        rendered = render_generation_prefix(tokenizer, conversations)
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=True).to(device)
        task_type = row["task_type"]
        task_id = CLAIM_TASK_ID if task_type == "claim_verification" else QUERY_TASK_ID
        labels = CLAIM_LABELS if task_id == CLAIM_TASK_ID else QUERY_LABELS
        prompt_end = torch.tensor([encoded["input_ids"].shape[1] - 1], dtype=torch.long, device=device)
        started = time.time()
        with torch.inference_mode():
            output = model(
                encoded["input_ids"],
                torch.tensor([task_id], dtype=torch.long, device=device),
                attention_mask=encoded["attention_mask"],
                prompt_end_indices=prompt_end,
                use_cache=False,
            )
            tensor = output.decisions.claim_probabilities[0] if task_id == CLAIM_TASK_ID else output.decisions.query_probabilities[0]
            probabilities = {label: float(tensor[position].item()) for position, label in enumerate(labels)}
            prediction = decode(task_type, probabilities, recipe["decoding"])
            forced_prefix = f"Decision: {prediction}\n"
            forced = tokenizer(rendered + forced_prefix, return_tensors="pt", add_special_tokens=True).to(device)
            continuation_budget = max(1, args.max_new_tokens - int(forced["input_ids"].shape[1] - encoded["input_ids"].shape[1]))
            generated = model.base_model.generate(
                input_ids=forced["input_ids"],
                attention_mask=forced["attention_mask"],
                max_new_tokens=continuation_budget,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )
        output_ids = generated[0, forced["input_ids"].shape[1]:]
        response = forced_prefix + clean_empty_think(tokenizer.decode(output_ids, skip_special_tokens=True)).strip()
        predictions.append(
            {
                "source_example_id": row.get("metadata", {}).get("source_example_id") or row.get("example_id"),
                "leakage_group_id": row.get("metadata", {}).get("leakage_group_id")
                or row.get("metadata", {}).get("source_group_id")
                or row.get("leakage_group_id")
                or row.get("metadata", {}).get("source_example_id")
                or row.get("example_id"),
                "task_type": task_type,
                "gold_label": row["gold_label"],
                "prediction": prediction,
                "decision_probabilities": probabilities,
                "response": response,
                "prompt_truncated": cap_info["truncated"],
                "latency_seconds": time.time() - started,
                "generated_tokens": int(output_ids.shape[0]),
                "evidence_condition": raw_condition,
                "reliability_condition": reliability_condition,
                "diagnostic_view": row.get("diagnostic_view", "full_input"),
                "weight_sha256": sha256_file(args.weight),
                "sealed_test": bool(freeze),
            }
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}", flush=True)
    write_jsonl(args.output, predictions)
    print(json.dumps({"status": "complete", "examples": len(predictions), "output": str(args.output), "output_sha256": sha256_file(args.output), "sealed_test": bool(freeze)}, indent=2))


if __name__ == "__main__":
    main()
