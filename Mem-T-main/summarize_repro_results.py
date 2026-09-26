#!/usr/bin/env python3
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("./Mem-T-main")


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_list(text: str):
    text = normalize_text(text)
    return text.split() if text else []


def f1_score(pred: str, gold: str) -> float:
    gtoks = token_list(gold)
    ptoks = token_list(pred)
    if not gtoks and not ptoks:
        return 1.0
    if not gtoks or not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    overlap = sum(min(pcount[t], gcount[t]) for t in pcount)
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(gtoks)
    return 2 * precision * recall / (precision + recall)


def bleu1_score(pred: str, gold: str) -> float:
    gtoks = token_list(gold)
    ptoks = token_list(pred)
    if not ptoks:
        return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks)
    if gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks) / len(ptoks))
    else:
        bp = 0.0
    return bp * precision


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def score_rows(rows):
    scored = []
    for row in rows:
        pred = row.get("pred", "")
        gold = row.get("gold", "")
        if isinstance(gold, list):
            f1 = max((f1_score(pred, g) for g in gold), default=0.0)
            bleu1 = max((bleu1_score(pred, g) for g in gold), default=0.0)
        else:
            f1 = f1_score(pred, str(gold))
            bleu1 = bleu1_score(pred, str(gold))
        row = dict(row)
        row["f1_score"] = f1
        row["bleu1_score"] = bleu1
        scored.append(row)
    return scored


def print_locomo():
    rows = load_jsonl(ROOT / "traj/locomo_pipeline_True_test_best/res.jsonl")
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category"))].append(row)
    print("[LoCoMo / Table 2]")
    for cat in sorted(by_cat, key=int):
        cat_rows = by_cat[cat]
        print(
            f"Category {cat}: n={len(cat_rows)}, "
            f"F1={mean([r['f1_score'] for r in cat_rows]):.6f}, "
            f"BLEU-1={mean([r['bleu1_score'] for r in cat_rows]):.6f}"
        )
    print(
        f"Overall: n={len(rows)}, "
        f"F1={mean([r['f1_score'] for r in rows]):.6f}, "
        f"BLEU-1={mean([r['bleu1_score'] for r in rows]):.6f}"
    )
    print()


def print_narrative_partial():
    files = [
        "traj/narrativeqa_local_summary_shard0of4_20260609_165718/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard1of4_20260609_165718/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard2of4_20260609_165718/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard3of4_20260609_165718/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard0of4_start372_20260612_180820/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard1of4_start344_20260612_180820/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard2of4_start423_20260612_180820/qa_trajectories.jsonl",
        "traj/narrativeqa_local_summary_shard3of4_start443_20260612_180820/qa_trajectories.jsonl",
    ]
    rows = []
    for rel in files:
        rows.extend(load_jsonl(ROOT / rel))
    rows = score_rows(rows)
    print("[NarrativeQA / Partial Summary Run]")
    print(
        f"Count={len(rows)}, "
        f"F1={mean([r['f1_score'] for r in rows]):.6f}, "
        f"BLEU-1={mean([r['bleu1_score'] for r in rows]):.6f}"
    )
    print("Note: this is a partial summary-based run, not directly comparable to Table 3.")
    print()


def print_hotpot_partial():
    files = [
        "traj/hotpotqa_local_shard0of4_20260612_182254/qa_trajectories.jsonl",
        "traj/hotpotqa_local_shard1of4_20260612_182254/qa_trajectories.jsonl",
        "traj/hotpotqa_local_shard2of4_20260612_182254/qa_trajectories.jsonl",
        "traj/hotpotqa_local_shard3of4_20260612_182254/qa_trajectories.jsonl",
    ]
    rows = []
    for rel in files:
        rows.extend(load_jsonl(ROOT / rel))
    rows = score_rows(rows)
    print("[HotpotQA / Partial]")
    print(
        f"Count={len(rows)}, "
        f"F1={mean([r['f1_score'] for r in rows]):.6f}, "
        f"BLEU-1={mean([r['bleu1_score'] for r in rows]):.6f}"
    )
    print("Note: only a few samples finished; this is not usable as the Table 3 final score.")
    print()


def print_longmemeval_oracle():
    rows = score_rows(load_jsonl(ROOT / "traj/longmemeval_oracle_k16_merged_20260612.jsonl"))
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[str(row.get("category", "unknown"))].append(row)
    print("[LongMemEval / Oracle Partial]")
    print(
        f"Count={len(rows)}, "
        f"F1={mean([r['f1_score'] for r in rows]):.6f}, "
        f"BLEU-1={mean([r['bleu1_score'] for r in rows]):.6f}"
    )
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        print(
            f"{cat}: n={len(cat_rows)}, "
            f"F1={mean([r['f1_score'] for r in cat_rows]):.6f}, "
            f"BLEU-1={mean([r['bleu1_score'] for r in cat_rows]):.6f}"
        )
    print("Note: Table 3 needs LLM-judge accuracy, which is still blocked by OpenAI API reachability.")
    print()


if __name__ == "__main__":
    print_locomo()
    print_hotpot_partial()
    print_longmemeval_oracle()
    print_narrative_partial()
