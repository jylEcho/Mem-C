import argparse
import concurrent.futures
import multiprocessing as mp
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
from dataset import load_locomo_dataset
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
    vector_db = VectorDBFactory.create_db(main_config.vector_db)
    llm_executor = LocalClient(model_name_or_path=main_config.llm.local_model_path, device=f"cuda:{gpu_id}")
    formation_module = MemoryFormation(llm_executor=llm_executor)
    update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
    retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=main_config)
    worker_builder = MemoryBuilder(vector_db=vector_db, formation=formation_module, update=update_module, config=main_config)
    worker_retrieval = retrieval_module
    worker_collector = get_collector(main_config.traj_dir)


def process_sample_logic(i, sample):
    sample_id = sample['qa'][0].get('sample_id') if sample['qa'] else f"sample_{i}"
    logger.info(f"Processing sample {sample_id}")
    for j, qa in enumerate(sample['qa']):
        result = worker_retrieval.retrieve_and_answer(qa['question'], sample_id=sample_id, category=qa.get('category', ''))
        worker_collector.log_qa_step(QATrajectoryLog(
            sample_id=sample_id,
            qa_id=f"{sample_id}_{j}",
            question=qa['question'],
            pred=result['answer'],
            gold=qa['answer'],
            evidence=qa['evidence'],
            traces=result['traces'],
            category=qa.get('category', ''),
        ))
    logger.info(f"Finished sample {sample_id}")


def process_sample_wrapper(args):
    return process_sample_logic(*args)


def build_config(workers, limit, name):
    config = SystemConfig()
    config.vector_db.db_type = "persistent"
    config.vector_db.path = "./Mem-T-main/database/locomo"
    config.vector_db.from_scratch = False
    config.data_name = "locomo"
    config.dataset_path = "./Mem-T-main/data/locomo/locomo10.json"
    config.mode = "test"
    config.USE_LOCAL_LLM = True
    config.USE_PARALLEL = True
    config.NUM_WORKERS = workers

    config.llm.local_model = f"./Code/Exp-A/merged_models/{name}"
    config.llm.local_model_path = f"./Code/Exp-A/merged_models/{name}"

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_limit{limit}" if limit else ""

    config.log_path = (
        f"./Code/Exp-A/logs/{name}/"
        f"locomo_local_{workers}w{suffix}_{stamp}.log"
    )
    config.traj_dir = (
        f"./Code/Exp-A/traj/{name}/"
        f"locomo_local_{workers}w{suffix}_{stamp}"
    )

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--name', type=str, default='locomo_rl_paper_8gpu_m2_n1_l1_k1')
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)
    config = build_config(workers=args.workers, limit=args.limit, name=args.name)
    seed_everything(seed=config.seed)
    logger.remove()
    logger.add(config.log_path, level="INFO", enqueue=True)

    chat_data = load_locomo_dataset(config.dataset_path)
    test_data = chat_data[2:]
    if args.limit:
        test_data = test_data[:args.limit]

    manager = mp.Manager()
    gpu_queue = manager.Queue()
    for i in range(config.NUM_WORKERS):
        gpu_queue.put(i % 8)

    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=config.NUM_WORKERS, initializer=init_worker, initargs=(gpu_queue, config)) as executor:
        futures = {executor.submit(process_sample_wrapper, (i, sample)): i for i, sample in enumerate(test_data)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            future.result()
            print(f"sample_done {idx}", flush=True)
    print(f"traj_dir={config.traj_dir}")
    print(f"elapsed_sec={time.time()-start:.1f}")

if __name__ == '__main__':
    main()
