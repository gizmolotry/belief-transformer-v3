#!/usr/bin/env python3
"""Extract complete paired contrast forward-pass logits and embeddings for all adversarial and disciplinary banks.

Evaluates on data/prospective_confirmatory_120_corpus.jsonl using microsoft/deberta-v2-xlarge-mnli.
Stores for each bank:
- la: [120, 8, 3] (logits for Hypothesis A)
- lb: [120, 8, 3] (logits for Hypothesis B)
- emb_a: [120, 8, hidden_dim] (mean-pooled embeddings for Hypothesis A)
- emb_b: [120, 8, hidden_dim] (mean-pooled embeddings for Hypothesis B)

Saves to data/cache_adversarial_kuhnian_logits.pt.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_prospective_corpus() -> List[Dict[str, Any]]:
    path = ROOT / "data" / "prospective_confirmatory_120_corpus.jsonl"
    articles = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def load_all_banks() -> Dict[str, Dict[str, Any]]:
    banks = {}

    # Q0 (Original Geopolitical)
    q0_path = ROOT / "config" / "framing_queries.yaml"
    with open(q0_path, "r", encoding="utf-8-sig") as f:
        q0_data = yaml.safe_load(f)
    banks["q0_original_geopolitical"] = {
        "name": "Geopolitical Conflict (Q0)",
        "pairs": [(v["A"], v["B"]) for v in q0_data["queries"].values()],
    }

    # Q1 (Paraphrase Control)
    q1_path = ROOT / "config" / "prompt_banks" / "probe_hypotheses_paraphrase_a.json"
    with open(q1_path, "r", encoding="utf-8-sig") as f:
        q1_raw = json.load(f)
    banks["q1_paraphrase_control"] = {
        "name": "Paraphrase Control (Q1)",
        "pairs": [(q1_raw[2 * i], q1_raw[2 * i + 1]) for i in range(8)],
    }

    # Q2 (Humanitarian)
    q2_path = ROOT / "config" / "prompt_banks" / "probe_hypotheses_q2_humanitarian.json"
    with open(q2_path, "r", encoding="utf-8-sig") as f:
        q2_raw = json.load(f)
    banks["q2_humanitarian_rights"] = {
        "name": "Humanitarian Rights (Q2)",
        "pairs": [(q2_raw[2 * i], q2_raw[2 * i + 1]) for i in range(8)],
    }

    # Q3 (Realpolitik)
    q3_path = ROOT / "config" / "prompt_banks" / "probe_hypotheses_q3_realpolitik.json"
    with open(q3_path, "r", encoding="utf-8-sig") as f:
        q3_raw = json.load(f)
    banks["q3_realpolitik_statecraft"] = {
        "name": "Realpolitik Statecraft (Q3)",
        "pairs": [(q3_raw[2 * i], q3_raw[2 * i + 1]) for i in range(8)],
    }

    # Adversarial and Disciplinary Banks
    adv_path = ROOT / "config" / "prompt_banks" / "adversarial_and_disciplinary_banks.json"
    with open(adv_path, "r", encoding="utf-8-sig") as f:
        adv_data = json.load(f)

    for cat_name, cat_dict in adv_data.items():
        for b_key, b_val in cat_dict.items():
            full_key = f"{cat_name}:{b_key}"
            banks[full_key] = {
                "name": b_val.get("name", b_key),
                "pairs": [(p[0], p[1]) for p in b_val["pairs"]],
            }

    return banks


def extract_bank_logits_and_embeddings(
    articles: List[Dict[str, Any]],
    pairs: List[Tuple[str, str]],
    model: Any,
    tokenizer: Any,
    device: str,
    batch_size: int = 16,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run forward passes for all 8 pairs over all articles on side A and side B."""
    N = len(articles)
    n_pairs = len(pairs)
    assert n_pairs == 8, f"Expected 8 pairs, got {n_pairs}"

    texts = [art.get("text", "") for art in articles]

    la_list = []
    lb_list = []
    emba_list = []
    embb_list = []

    for pair_idx, (hyp_a, hyp_b) in enumerate(pairs):
        # Process Hypothesis A
        pair_la = []
        pair_emba = []
        for start_idx in range(0, N, batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            inputs_a = tokenizer(
                batch_texts,
                [hyp_a] * len(batch_texts),
                return_tensors="pt",
                truncation="only_first",
                max_length=max_length,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs_a = model(**inputs_a, output_hidden_states=True, return_dict=True)
                logits_a = outputs_a.logits.cpu().to(torch.float16)
                last_hidden_a = outputs_a.hidden_states[-1]
                mask_a = inputs_a["attention_mask"].unsqueeze(-1).expand(last_hidden_a.size()).float()
                sum_emb_a = torch.sum(last_hidden_a * mask_a, 1)
                sum_mask_a = torch.clamp(mask_a.sum(1), min=1e-9)
                mean_emb_a = (sum_emb_a / sum_mask_a).cpu().to(torch.float16)

            pair_la.append(logits_a)
            pair_emba.append(mean_emb_a)

        la_list.append(torch.cat(pair_la, dim=0))  # [N, 3]
        emba_list.append(torch.cat(pair_emba, dim=0))  # [N, hidden]

        # Process Hypothesis B
        pair_lb = []
        pair_embb = []
        for start_idx in range(0, N, batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            inputs_b = tokenizer(
                batch_texts,
                [hyp_b] * len(batch_texts),
                return_tensors="pt",
                truncation="only_first",
                max_length=max_length,
                padding=True,
            ).to(device)

            with torch.no_grad():
                outputs_b = model(**inputs_b, output_hidden_states=True, return_dict=True)
                logits_b = outputs_b.logits.cpu().to(torch.float16)
                last_hidden_b = outputs_b.hidden_states[-1]
                mask_b = inputs_b["attention_mask"].unsqueeze(-1).expand(last_hidden_b.size()).float()
                sum_emb_b = torch.sum(last_hidden_b * mask_b, 1)
                sum_mask_b = torch.clamp(mask_b.sum(1), min=1e-9)
                mean_emb_b = (sum_emb_b / sum_mask_b).cpu().to(torch.float16)

            pair_lb.append(logits_b)
            pair_embb.append(mean_emb_b)

        lb_list.append(torch.cat(pair_lb, dim=0))  # [N, 3]
        embb_list.append(torch.cat(pair_embb, dim=0))  # [N, hidden]

    # Stack along pair dimension: [N, 8, 3] and [N, 8, hidden]
    la_tensor = torch.stack(la_list, dim=1)
    lb_tensor = torch.stack(lb_list, dim=1)
    emba_tensor = torch.stack(emba_list, dim=1)
    embb_tensor = torch.stack(embb_list, dim=1)

    return la_tensor, lb_tensor, emba_tensor, embb_tensor


def main():
    print("=" * 80)
    print("EXTRACTION OF COMPLETE PAIRED CONTRAST FOR ADVERSARIAL KUHNIAN BATTERY")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ENV] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    articles = load_prospective_corpus()
    print(f"[DATA] Loaded {len(articles)} prospective confirmatory articles.")

    banks = load_all_banks()
    print(f"[BANKS] Loaded {len(banks)} total hypothesis banks for evaluation.")

    # Check for existing legacy cache to verify Q0/Q1 numerical parity
    legacy_path = ROOT / "data" / "cache_prospective_120_q0_q1_logits.pt"
    legacy_cache = {}
    if legacy_path.exists():
        legacy_cache = torch.load(legacy_path, map_location="cpu", weights_only=False)
        print(f"[CACHE] Loaded legacy Q0/Q1 cache for numerical parity checks.")

    output_path = ROOT / "data" / "cache_adversarial_kuhnian_logits.pt"
    final_cache = {}
    if output_path.exists():
        try:
            final_cache = torch.load(output_path, map_location="cpu", weights_only=False)
            print(f"[CACHE] Loaded existing cache with {len(final_cache)} banks.")
        except Exception as e:
            print(f"[CACHE] Could not load existing cache: {e}. Starting fresh.")
            final_cache = {}

    model_name = "microsoft/deberta-v2-xlarge-mnli"
    print(f"[MODEL] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    total_start = time.time()
    for b_idx, (b_key, b_info) in enumerate(banks.items(), 1):
        if b_key in final_cache:
            print(f"[{b_idx}/{len(banks)}] Bank '{b_key}' already cached. Skipping.")
            continue

        print(f"\n[{b_idx}/{len(banks)}] Extracting bank: '{b_key}' ({b_info['name']})...")
        t0 = time.time()
        la, lb, emba, embb = extract_bank_logits_and_embeddings(
            articles, b_info["pairs"], model, tokenizer, device, batch_size=16
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s! Logits shape: {la.shape}, Embeddings shape: {emba.shape}")

        final_cache[b_key] = {
            "name": b_info["name"],
            "la": la,
            "lb": lb,
            "emb_a": emba,
            "emb_b": embb,
        }

        # Check parity against legacy cache if available
        if b_key == "q0_original_geopolitical" and "q0_la" in legacy_cache:
            diff_a = torch.max(torch.abs(la.float() - legacy_cache["q0_la"].float())).item()
            diff_b = torch.max(torch.abs(lb.float() - legacy_cache["q0_lb"].float())).item()
            print(f"  [PARITY CHECK Q0] Max diff LA: {diff_a:.6f}, Max diff LB: {diff_b:.6f}")
        elif b_key == "q1_paraphrase_control" and "q1_la" in legacy_cache:
            diff_a = torch.max(torch.abs(la.float() - legacy_cache["q1_la"].float())).item()
            diff_b = torch.max(torch.abs(lb.float() - legacy_cache["q1_lb"].float())).item()
            print(f"  [PARITY CHECK Q1] Max diff LA: {diff_a:.6f}, Max diff LB: {diff_b:.6f}")

        # Checkpoint to disk after each bank
        torch.save(final_cache, output_path)
        print(f"  [CHECKPOINT] Saved cache with {len(final_cache)} banks to {output_path}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"EXTRACTION COMPLETE! Total elapsed time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print(f"Cache saved to {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
