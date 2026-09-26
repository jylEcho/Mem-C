"""Create deterministic, disjoint LoCoMo subsets for the Exp-B screening study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("./external/Mem-T")
SOURCE = ROOT / "Code/ChatMem-main/reproduction/runs/20260822_full_reproduction/paper_protocol/locomo_v1/tree_grpo"
OUT = ROOT / "Code/Exp-B/reporting-v1/data"
SEED = 20260830


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: dict[str, object] = {"selection_seed": SEED, "source": str(SOURCE), "splits": {}}
    for split in ("train", "valid"):
        source = SOURCE / f"{split}.parquet"
        frame = pd.read_parquet(source)
        subset = frame.sample(n=16, random_state=SEED).sort_values("sample_id").reset_index(drop=True)
        destination = OUT / f"{split}.parquet"
        subset.to_parquet(destination, index=False)
        records["splits"][split] = {
            "source_rows": len(frame),
            "subset_rows": len(subset),
            "sample_ids": subset["sample_id"].tolist(),
            "sha256": fingerprint(destination),
        }
    (OUT / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
