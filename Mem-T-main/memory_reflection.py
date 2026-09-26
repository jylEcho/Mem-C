from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Dict, List


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _default_confidence(memory_kind: str) -> float:
    mapping = {
        "summary": 0.78,
        "persona": 0.74,
        "fact": 0.72,
        "experience": 0.68,
        "turn": 0.66,
    }
    return mapping.get(memory_kind, 0.70)


def build_reliability_metadata(
    base_metadata: Dict[str, Any] | None,
    memory_kind: str,
    source_turn_ids: List[Any] | None = None,
    existing_metadata: Dict[str, Any] | None = None,
    action: str = "add",
) -> Dict[str, Any]:
    meta: Dict[str, Any] = deepcopy(existing_metadata or {})
    if base_metadata:
        meta.update(base_metadata)

    support_count = max(
        _to_int(meta.get("support_count"), 0),
        len(source_turn_ids or []),
    )
    contradiction_count = _to_int(meta.get("contradiction_count"), 0)
    verification_count = _to_int(meta.get("verification_count"), 0)
    repair_count = _to_int(meta.get("repair_count"), 0)
    usage_count = _to_int(meta.get("usage_count"), 0)
    success_credit = _to_float(meta.get("success_credit"), 0.0)
    failure_credit = _to_float(meta.get("failure_credit"), 0.0)

    if action in {"update", "revise"}:
        repair_count += 1
        verification_count += 1
    elif action == "reinforce":
        support_count += 1
        verification_count += 1
    elif action == "add":
        verification_count = max(verification_count, 1)
    elif action == "verify":
        verification_count += 1
    elif action == "credit_positive":
        usage_count += 1
        success_credit += 1.0
        verification_count += 1
    elif action == "credit_negative":
        usage_count += 1
        failure_credit += 1.0
        contradiction_count += 1
    elif action == "quarantine":
        contradiction_count = max(contradiction_count, 1)
        repair_count += 1

    confidence = _to_float(meta.get("confidence"), _default_confidence(memory_kind))
    confidence += min(0.08, 0.01 * verification_count)
    confidence += min(0.05, 0.005 * support_count)
    confidence -= min(0.18, 0.04 * contradiction_count)
    confidence -= min(0.10, 0.02 * repair_count)
    confidence += min(0.12, 0.02 * success_credit)
    confidence -= min(0.18, 0.03 * failure_credit)
    confidence = max(0.05, min(0.99, confidence))

    causal_credit = confidence + min(0.08, 0.01 * support_count)
    causal_credit += min(0.05, 0.01 * verification_count)
    causal_credit += min(0.15, 0.03 * success_credit)
    causal_credit -= min(0.20, 0.04 * failure_credit)
    causal_credit -= min(0.20, 0.05 * contradiction_count)
    causal_credit = max(0.0, min(1.0, causal_credit))

    previous_status = meta.get("memory_status", "active")
    memory_status = previous_status
    if action == "quarantine":
        memory_status = "quarantined"
    elif previous_status == "quarantined" and action not in {"revise", "reinforce", "update"}:
        memory_status = "quarantined"
    elif contradiction_count >= 3 and confidence < 0.40:
        memory_status = "quarantined"
    elif contradiction_count > 0 and confidence < 0.6:
        memory_status = "conflicted"
    elif failure_credit > success_credit and failure_credit >= 2:
        memory_status = "stale"
    else:
        memory_status = "active"

    meta.update(
        {
            "memory_kind": memory_kind,
            "memory_status": memory_status,
            "confidence": round(confidence, 4),
            "causal_credit": round(causal_credit, 4),
            "support_count": support_count,
            "verification_count": verification_count,
            "repair_count": repair_count,
            "contradiction_count": contradiction_count,
            "usage_count": usage_count,
            "success_credit": round(success_credit, 4),
            "failure_credit": round(failure_credit, 4),
            "last_refreshed_time": _now_str(),
        }
    )
    if action in {"update", "revise", "reinforce", "quarantine"}:
        meta["last_repair_time"] = _now_str()
    if action in {"verify", "reinforce", "credit_positive", "credit_negative"}:
        meta["last_verified_time"] = _now_str()
    return meta


def reliability_score(metadata: Dict[str, Any] | None) -> float:
    if not metadata:
        return 0.0
    score = _to_float(metadata.get("confidence"), 0.0)
    score += 0.05 * _to_float(metadata.get("causal_credit"), 0.0)
    score += 0.01 * _to_int(metadata.get("support_count"), 0)
    score += 0.005 * _to_int(metadata.get("verification_count"), 0)
    score += 0.03 * _to_float(metadata.get("success_credit"), 0.0)
    score -= 0.04 * _to_float(metadata.get("failure_credit"), 0.0)
    score -= 0.03 * _to_int(metadata.get("repair_count"), 0)
    score -= 0.06 * _to_int(metadata.get("contradiction_count"), 0)
    if metadata.get("memory_status") == "quarantined":
        score -= 1.0
    return score


def utility_score(metadata: Dict[str, Any] | None) -> float:
    if not metadata:
        return 0.0
    score = 0.02 * _to_int(metadata.get("usage_count"), 0)
    score += 0.08 * _to_float(metadata.get("success_credit"), 0.0)
    score -= 0.06 * _to_float(metadata.get("failure_credit"), 0.0)
    if metadata.get("memory_status") == "stale":
        score -= 0.1
    if metadata.get("memory_status") == "quarantined":
        score -= 0.5
    return score


def combined_rank_score(metadata: Dict[str, Any] | None, distance: Any = None) -> float:
    relevance = 0.5
    if distance is not None:
        dist = _to_float(distance, 1.0)
        relevance = max(0.0, min(1.0, 1.0 - dist))
    return 0.45 * relevance + 0.35 * reliability_score(metadata) + 0.20 * utility_score(metadata)


def rank_search_results(results: Dict[str, Any]) -> Dict[str, Any]:
    if os.getenv("MEMT_DISABLE_REFLECTIVE", "0") == "1":
        return results
    if not results or not results.get("documents") or not results.get("metadatas"):
        return results

    docs_batch = results.get("documents", [])
    metas_batch = results.get("metadatas", [])
    ids_batch = results.get("ids", [])
    distances_batch = results.get("distances", [])

    ranked_docs = []
    ranked_metas = []
    ranked_ids = []
    ranked_distances = []

    for batch_idx, docs in enumerate(docs_batch):
        metas = metas_batch[batch_idx] if batch_idx < len(metas_batch) else []
        ids = ids_batch[batch_idx] if batch_idx < len(ids_batch) else []
        distances = distances_batch[batch_idx] if batch_idx < len(distances_batch) else []
        rows = list(zip(docs, metas, ids or [None] * len(docs), distances or [None] * len(docs)))
        rows.sort(key=lambda row: combined_rank_score(row[1], row[3]), reverse=True)
        ranked_docs.append([row[0] for row in rows])
        ranked_metas.append([row[1] for row in rows])
        ranked_ids.append([row[2] for row in rows])
        ranked_distances.append([row[3] for row in rows])

    results["documents"] = ranked_docs
    results["metadatas"] = ranked_metas
    if ids_batch:
        results["ids"] = ranked_ids
    if distances_batch:
        results["distances"] = ranked_distances
    return results


def format_reliability_tag(metadata: Dict[str, Any] | None) -> str:
    if os.getenv("MEMT_DISABLE_REFLECTIVE", "0") == "1":
        return ""
    if not metadata:
        return ""
    return (
        f"[status={metadata.get('memory_status', 'active')}; "
        f"confidence={_to_float(metadata.get('confidence'), 0.0):.2f}; "
        f"credit={_to_float(metadata.get('causal_credit'), 0.0):.2f}; "
        f"success={_to_float(metadata.get('success_credit'), 0.0):.1f}; "
        f"failure={_to_float(metadata.get('failure_credit'), 0.0):.1f}]"
    )
