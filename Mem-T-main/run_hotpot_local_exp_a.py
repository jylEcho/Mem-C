import argparse
import concurrent.futures
import multiprocessing as mp
import os
import sys
import time
import types
from datetime import datetime

fake_vllm = types.ModuleType("vllm")


class _Dummy:
    pass


fake_vllm.LLM = _Dummy
fake_vllm.SamplingParams = _Dummy
sys.modules.setdefault("vllm", fake_vllm)

from loguru import logger

from config import SystemConfig
from dataset import load_hotpotqa_dataset, train_valid_test_split
from llm_api import LocalClient
from memory_builder import MemoryBuilder
from memory_formation import MemoryFormation
from memory_retrieval import MemoryRetriever
from memory_update import MemoryUpdate
from trajectory_logger import QATrajectoryLog, get_collector
from utils import seed_everything
from vector_db import VectorDBFactory


worker_builder = None
worker_retrieval = None
worker_collector = None
main_config = None


def init_worker(gpu_queue, config_from_main):
    global worker_builder, worker_retrieval, worker_collector, main_config

    main_config = config_from_main

    logger.remove()
    logger.add(main_config.log_path, level="INFO", enqueue=True)

    try:
        gpu_id = gpu_queue.get(timeout=5)
    except Exception:
        gpu_id = 0

    # HotpotQA 需要动态构建 memory，并且 from_scratch=True。
    # 不同 worker 不能同时操作同一个 persistent DB，
    # 因此每个 worker 使用独立数据库。
    base_db_path = main_config.vector_db.path
    main_config.vector_db.path = f"{base_db_path}_worker_{os.getpid()}"

    logger.info(
        f"Worker initialized: pid={os.getpid()}, "
        f"gpu=cuda:{gpu_id}, db={main_config.vector_db.path}"
    )

    vector_db = VectorDBFactory.create_db(main_config.vector_db)

    llm_executor = LocalClient(
        model_name_or_path=main_config.llm.local_model_path,
        device=f"cuda:{gpu_id}",
    )

    formation_module = MemoryFormation(
        llm_executor=llm_executor
    )

    update_module = MemoryUpdate(
        llm_executor=llm_executor,
        vector_db=vector_db,
    )

    retrieval_module = MemoryRetriever(
        llm_executor=llm_executor,
        vector_db=vector_db,
        config=main_config,
    )

    worker_builder = MemoryBuilder(
        vector_db=vector_db,
        formation=formation_module,
        update=update_module,
        config=main_config,
    )

    worker_retrieval = retrieval_module
    worker_collector = get_collector(main_config.traj_dir)


def process_sample_logic(i, sample):
    if sample["qa"]:
        sample_id = sample["qa"][0].get(
            "sample_id",
            f"sample_{i}",
        )
    else:
        sample_id = f"sample_{i}"

    logger.info(f"Processing sample {sample_id}")

    # HotpotQA 与 LoCoMo 的关键区别：
    # 每个 sample 需要先构建 memory
    worker_builder.build_from_sample(sample)

    for j, qa in enumerate(sample["qa"]):
        result = worker_retrieval.retrieve_and_answer(
            qa["question"],
            sample_id=sample_id,
            category=qa.get("category", ""),
        )

        worker_collector.log_qa_step(
            QATrajectoryLog(
                sample_id=sample_id,
                qa_id=f"{sample_id}_{j}",
                question=qa["question"],
                pred=result["answer"],
                gold=qa["answer"],
                evidence=qa["evidence"],
                traces=result["traces"],
                category=qa.get("category", ""),
            )
        )

    logger.info(f"Finished sample {sample_id}")


def process_sample_wrapper(args):
    return process_sample_logic(*args)


def build_config(workers, limit, name):
    config = SystemConfig()

    config.vector_db.db_type = "persistent"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 这是 DB 的基础路径。
    # 每个 worker 启动后会自动追加 worker pid。
    config.vector_db.path = (
        f"./Code/Exp-A/database/{name}/"
        f"hotpotqa_{stamp}"
    )

    config.vector_db.from_scratch = True

    config.data_name = "hotpotqa"
    config.dataset_path = (
        "./Mem-T-main/"
        "data/hotpotqa/eval_400.json"
    )

    config.mode = "test"

    config.USE_LOCAL_LLM = True
    config.USE_PARALLEL = True
    config.NUM_WORKERS = workers

    config.llm.local_model = (
        f"./Code/Exp-A/"
        f"merged_models/{name}"
    )

    config.llm.local_model_path = (
        f"./Code/Exp-A/"
        f"merged_models/{name}"
    )

    suffix = f"_limit{limit}" if limit else ""

    config.log_path = (
        f"./Code/Exp-A/logs/{name}/"
        f"hotpotqa_local_{workers}w{suffix}_{stamp}.log"
    )

    config.traj_dir = (
        f"./Code/Exp-A/traj/{name}/"
        f"hotpotqa_local_{workers}w{suffix}_{stamp}"
    )

    return config


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--name",
        type=str,
        default="locomo_rl_paper_8gpu_m2_n1_l1_k1",
    )

    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    config = build_config(
        workers=args.workers,
        limit=args.limit,
        name=args.name,
    )

    seed_everything(seed=config.seed)

    logger.remove()
    logger.add(
        config.log_path,
        level="INFO",
        enqueue=True,
    )

    chat_data = load_hotpotqa_dataset(
        config.dataset_path
    )

    train_data, valid_data, test_data = train_valid_test_split(
        chat_data,
        seed=config.seed,
    )

    # 保留原来的 72 个测试样本设置
    test_data = test_data[:72]

    if args.limit:
        test_data = test_data[:args.limit]

    manager = mp.Manager()
    gpu_queue = manager.Queue()

    # 8 GPU round-robin
    for i in range(config.NUM_WORKERS):
        gpu_queue.put(i % 8)

    start = time.time()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=config.NUM_WORKERS,
        initializer=init_worker,
        initargs=(gpu_queue, config),
    ) as executor:

        futures = {
            executor.submit(
                process_sample_wrapper,
                (i, sample),
            ): i
            for i, sample in enumerate(test_data)
        }

        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]

            # 如果 worker 报错，这里直接抛出，
            # 避免 silent failure。
            future.result()

            print(
                f"sample_done {idx}",
                flush=True,
            )

    print(f"traj_dir={config.traj_dir}")
    print(f"db_base_path={config.vector_db.path}")
    print(f"elapsed_sec={time.time() - start:.1f}")


if __name__ == "__main__":
    main()