import argparse
import os
import re
import sys
import time
from datetime import datetime
import types
from datasets import load_dataset

fake_vllm = types.ModuleType("vllm")
class _Dummy:
    pass
fake_vllm.LLM = _Dummy
fake_vllm.SamplingParams = _Dummy
sys.modules.setdefault("vllm", fake_vllm)

from loguru import logger

from config import SystemConfig
from dataset import load_locomo_dataset, load_hotpotqa_dataset, load_longmemeval_dataset
from llm_api import LocalClient, VLLMClient
from memory_builder import MemoryBuilder
from memory_formation import MemoryFormation
from memory_retrieval import MemoryRetriever
from memory_update import MemoryUpdate
from trajectory_logger import QATrajectoryLog, get_collector
from utils import seed_everything
from vector_db import VectorDBFactory


def chunk_text(text, max_chars):
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            split = text.rfind("\n", start, end)
            if split <= start:
                split = text.rfind(" ", start, end)
            if split > start:
                end = split
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


def group_paragraphs(paragraphs, paragraphs_per_session, max_chars_per_session):
    groups = []
    current = []
    current_chars = 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)
        too_many_paras = current and len(current) >= paragraphs_per_session
        too_many_chars = current and (current_chars + para_len + 2 > max_chars_per_session)
        if too_many_paras or too_many_chars:
            groups.append("\n\n".join(current))
            current = []
            current_chars = 0
        if para_len > max_chars_per_session:
            if current:
                groups.append("\n\n".join(current))
                current = []
                current_chars = 0
            groups.extend(chunk_text(para, max_chars_per_session))
            continue
        current.append(para)
        current_chars += para_len + 2
    if current:
        groups.append("\n\n".join(current))
    return groups


def load_narrativeqa_custom(filepath, source="summary", paragraphs_per_session=32, max_chars_per_session=24000):
    logger.info(f"Loading NarrativeQA from {filepath} with source={source}")
    try:
        dataset = load_dataset(filepath)
    except Exception:
        dataset = load_dataset("parquet", data_files={"test": os.path.join(filepath, "data/test-*.parquet")})

    test_split = dataset["test"]
    chat_dataset = []
    for idx, sample in enumerate(test_split):
        doc = sample.get("document", {}) or {}
        q = sample.get("question", {}) or {}
        answers = sample.get("answers", [])
        sample_id = str(doc.get("id", idx)) + f"_{idx}"
        question = q.get("text", "") if isinstance(q, dict) else str(q)

        if hasattr(answers, "tolist"):
            answers = answers.tolist()
        answer_list = []
        for ans in answers:
            if isinstance(ans, dict):
                answer_list.append(ans.get("text", ""))
            else:
                answer_list.append(str(ans))
        answer_list = [a for a in answer_list if a]
        if not answer_list:
            answer_list = ["I don't know."]

        if source == "summary":
            summary_obj = doc.get("summary", {}) if isinstance(doc, dict) else {}
            if isinstance(summary_obj, dict):
                merged = "\n\n".join(
                    [part.strip() for part in [summary_obj.get("title", ""), summary_obj.get("text", "")] if part and str(part).strip()]
                )
                raw_segments = [merged] if merged else []
            else:
                raw_segments = [str(summary_obj)]
            segments = [seg.strip() for seg in raw_segments if seg and str(seg).strip()]
            sessions_text = []
            for seg in segments:
                sessions_text.extend(chunk_text(seg, max_chars_per_session))
            if not sessions_text:
                sessions_text = [""]
        else:
            text_content = doc.get("text", "") if isinstance(doc, dict) else str(doc)
            paragraphs = [seg.strip() for seg in re.split(r"\n\s*\n+", text_content) if seg.strip()]
            if not paragraphs:
                paragraphs = chunk_text(text_content[: max_chars_per_session * 4], max_chars_per_session)
            sessions_text = group_paragraphs(paragraphs, paragraphs_per_session, max_chars_per_session)
            if not sessions_text:
                sessions_text = [""]

        conversation_id = f"{sample_id}_conv"
        formatted_conversation = []
        for i, text in enumerate(sessions_text):
            session_id = f"{conversation_id}_s{i}"
            formatted_conversation.append(
                {
                    "session_id": session_id,
                    "session_turns": [
                        {
                            "turn_id": f"{session_id}_t0",
                            "speaker": "document",
                            "text": text,
                        }
                    ],
                    "metadata": {
                        "sample_id": sample_id,
                        "conversation_id": conversation_id,
                        "session_id": session_id,
                        "title": doc.get("url", "") if isinstance(doc, dict) else "",
                        "speaker_a": "document",
                        "speaker_b": "document",
                    },
                }
            )

        formatted_qa = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer_list,
            "evidence": [],
            "category": f"narrativeqa_{source}",
        }
        chat_dataset.append({"qa": [formatted_qa], "conversation": formatted_conversation})

    logger.info(f"{len(chat_dataset)} NarrativeQA samples loaded")
    return chat_dataset


def build_config(args):
    config = SystemConfig()
    config.vector_db.db_type = args.db_type
    if args.db_type == "http":
        config.vector_db.host = args.db_host
        config.vector_db.port = args.db_port
        if args.db_path:
            config.vector_db.path = args.db_path
    elif args.db_path:
        config.vector_db.path = args.db_path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    extra_tags = []
    if args.data_name == "longmemeval":
        extra_tags.append(args.longmemeval_variant.replace("longmemeval_", ""))
    if args.data_name == "narrativeqa":
        extra_tags.append(args.narrativeqa_source)
        if args.narrativeqa_source == "text":
            extra_tags.append(f"p{args.narrativeqa_paragraphs_per_session}")
            extra_tags.append(f"c{args.narrativeqa_max_chars_per_session}")
    if args.summary_context_turns != 4:
        extra_tags.append(f"k{args.summary_context_turns}")
    if args.device != "cuda:0":
        extra_tags.append(args.device.replace(":", ""))
    if args.limit:
        extra_tags.append(f"limit{args.limit}")
    if args.num_shards > 1:
        extra_tags.append(f"shard{args.shard_id}of{args.num_shards}")
    if args.start_index:
        extra_tags.append(f"start{args.start_index}")
    suffix = ("_" + "_".join(extra_tags)) if extra_tags else ""

    if args.db_type == "persistent":
        config.vector_db.path = args.db_path or f"./database/{args.data_name}_local{suffix}_{stamp}"
    config.vector_db.from_scratch = True
    config.data_name = args.data_name
    if args.data_name == "locomo":
        config.dataset_path = "./data/locomo/locomo10.json"
    elif args.data_name == "hotpotqa":
        config.dataset_path = "./data/hotpotqa/eval_400.json"
    elif args.data_name == "longmemeval":
        config.dataset_path = f"./data/longmemeval/{args.longmemeval_variant}.json"
    elif args.data_name == "narrativeqa":
        config.dataset_path = "./data/narrativeqa"
    else:
        raise ValueError(args.data_name)

    config.mode = "test"
    config.USE_LOCAL_LLM = True
    config.USE_PARALLEL = False
    config.NUM_WORKERS = 1
    config.memory.summary_context_turns = args.summary_context_turns
    # config.llm.local_model = "models/Mem-T-4B"
    # config.llm.local_model_path = "models/Mem-T-4B"
    config.llm.local_model = args.model_path
    config.llm.local_model_path = args.model_path
    config.log_path = f"./Code/Exp-A/logs/{args.run_name}/{args.data_name}_local{suffix}_{stamp}.log"
    config.traj_dir = f"./Code/Exp-A/traj/{args.run_name}/{args.data_name}_local{suffix}_{stamp}"
    return config


def load_data(config, args):
    if config.data_name == "locomo":
        data = load_locomo_dataset(config.dataset_path)[2:]
    elif config.data_name == "hotpotqa":
        data = load_hotpotqa_dataset(config.dataset_path)
    elif config.data_name == "longmemeval":
        data = load_longmemeval_dataset(config.dataset_path)
    elif config.data_name == "narrativeqa":
        data = load_narrativeqa_custom(
            config.dataset_path,
            source=args.narrativeqa_source,
            paragraphs_per_session=args.narrativeqa_paragraphs_per_session,
            max_chars_per_session=args.narrativeqa_max_chars_per_session,
        )
    else:
        raise ValueError(config.data_name)
    if args.num_shards > 1:
        data = data[args.shard_id :: args.num_shards]
    if args.start_index:
        data = data[args.start_index :]
    if args.limit:
        data = data[: args.limit]
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", required=True, choices=["locomo", "hotpotqa", "longmemeval", "narrativeqa"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to local model",
    )
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--summary_context_turns", type=int, default=4)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--db_type", choices=["persistent", "http"], default="persistent")
    parser.add_argument("--db_host", default="localhost")
    parser.add_argument("--db_port", type=int, default=8070)
    parser.add_argument("--db_path", default="")
    parser.add_argument(
        "--longmemeval_variant",
        default="longmemeval_s_cleaned",
        choices=["longmemeval_s_cleaned", "longmemeval_m_cleaned", "longmemeval_oracle"],
    )
    parser.add_argument("--narrativeqa_source", default="summary", choices=["summary", "text"])
    parser.add_argument("--narrativeqa_paragraphs_per_session", type=int, default=32)
    parser.add_argument("--narrativeqa_max_chars_per_session", type=int, default=24000)
    args = parser.parse_args()
    if args.shard_id < 0 or args.num_shards < 1 or args.shard_id >= args.num_shards:
        raise ValueError("Require 0 <= shard_id < num_shards and num_shards >= 1")
    if args.start_index < 0:
        raise ValueError("Require start_index >= 0")

    config = build_config(args)
    seed_everything(seed=config.seed)
    logger.remove()
    logger.add(config.log_path, level="INFO", enqueue=True)

    data = load_data(config, args)
    print(
        f"dataset={args.data_name} samples={len(data)} device={args.device} shard={args.shard_id}/{args.num_shards} start={args.start_index}",
        flush=True,
    )
    print(f"dataset_path={config.dataset_path}", flush=True)
    print(f"traj_dir={config.traj_dir}", flush=True)
    print(f"db_path={config.vector_db.path}", flush=True)

    vector_db = VectorDBFactory.create_db(config.vector_db)
    llm_executor = LocalClient(model_name_or_path=config.llm.local_model_path, device=args.device)
    formation_module = MemoryFormation(llm_executor=llm_executor)
    update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
    retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=config)
    builder = MemoryBuilder(vector_db=vector_db, formation=formation_module, update=update_module, config=config)
    collector = get_collector(config.traj_dir)

    start = time.time()
    for i, sample in enumerate(data):
        sample_id = sample["qa"][0].get("sample_id", f"sample_{i}")
        print(f"building {i} {sample_id}", flush=True)
        builder.build_from_sample(sample)
        for j, qa in enumerate(sample["qa"]):
            result = retrieval_module.retrieve_and_answer(
                qa["question"], sample_id=sample_id, category=qa.get("category", "")
            )
            collector.log_qa_step(
                QATrajectoryLog(
                    sample_id=sample_id,
                    qa_id=f"{sample_id}_{j}",
                    question=qa["question"],
                    pred=result["answer"],
                    gold=qa["answer"],
                    evidence=qa.get("evidence", []),
                    traces=result["traces"],
                    category=qa.get("category", ""),
                )
            )
        print(f"sample_done {i} {sample_id}", flush=True)

    print(f"elapsed_sec={time.time() - start:.1f}")


if __name__ == "__main__":
    main()
