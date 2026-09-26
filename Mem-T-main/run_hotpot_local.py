import argparse
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


def build_config(limit):
    config = SystemConfig()
    config.vector_db.db_type = "persistent"
    config.vector_db.path = f"./database/hotpotqa_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config.vector_db.from_scratch = True
    config.data_name = "hotpotqa"
    config.dataset_path = "./data/hotpotqa/eval_400.json"
    config.mode = "test"
    config.USE_LOCAL_LLM = True
    config.USE_PARALLEL = False
    config.NUM_WORKERS = 1
    config.llm.local_model = "models/Mem-T-4B"
    config.llm.local_model_path = "models/Mem-T-4B"
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f"_limit{limit}" if limit else ""
    config.log_path = f"./logs/hotpotqa_local{suffix}_{stamp}.log"
    config.traj_dir = f"./traj/hotpotqa_local{suffix}_{stamp}"
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    config = build_config(args.limit)
    seed_everything(seed=config.seed)
    logger.remove()
    logger.add(config.log_path, level="INFO", enqueue=True)

    chat_data = load_hotpotqa_dataset(config.dataset_path)
    train_data, valid_data, test_data = train_valid_test_split(chat_data, seed=config.seed)
    test_data = test_data[:72]
    if args.limit:
        test_data = test_data[:args.limit]

    vector_db = VectorDBFactory.create_db(config.vector_db)
    llm_executor = LocalClient(model_name_or_path=config.llm.local_model_path, device="cuda:0")
    formation_module = MemoryFormation(llm_executor=llm_executor)
    update_module = MemoryUpdate(llm_executor=llm_executor, vector_db=vector_db)
    retrieval_module = MemoryRetriever(llm_executor=llm_executor, vector_db=vector_db, config=config)
    builder = MemoryBuilder(vector_db=vector_db, formation=formation_module, update=update_module, config=config)
    collector = get_collector(config.traj_dir)

    start = time.time()
    for i, sample in enumerate(test_data):
        sample_id = sample['qa'][0].get('sample_id', f'sample_{i}')
        print(f"building {i} {sample_id}", flush=True)
        builder.build_from_sample(sample)
        for j, qa in enumerate(sample['qa']):
            result = retrieval_module.retrieve_and_answer(qa['question'], sample_id=sample_id, category=qa.get('category', ''))
            collector.log_qa_step(QATrajectoryLog(
                sample_id=sample_id,
                qa_id=f"{sample_id}_{j}",
                question=qa['question'],
                pred=result['answer'],
                gold=qa['answer'],
                evidence=qa['evidence'],
                traces=result['traces'],
                category=qa.get('category', ''),
            ))
        print(f"sample_done {i} {sample_id}", flush=True)

    print(f"traj_dir={config.traj_dir}")
    print(f"db_path={config.vector_db.path}")
    print(f"elapsed_sec={time.time()-start:.1f}")

if __name__ == '__main__':
    main()
