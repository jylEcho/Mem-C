# trajectory_logger.py
import json
import os
import time
import threading
from typing import Dict, Any, List
from dataclasses import dataclass, field
from loguru import logger

@dataclass
class MemTrajectoryLog:
    """用于记录单次完整的交互轨迹"""
    sample_id: str
    phase: str # 'formation', 'update', 'retrieval'
    op_id: str # unique id for this operation
    input_context: Any # Prompt or Context
    llm_response: str # Raw LLM Output
    session_id: str =""# Unique session identifier
    benchmark_name: str = "" # Benchmark name for management
    parsed_output: Any = ""# Parsed Tool Calls
    source_turn_ids: List[str] = field(default_factory=list) # IDs of turns that were the source for this step
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., source_turn_ids, reward metrics placeholders

@dataclass
class QATrajectoryLog:
    """用于记录单次完整的交互轨迹"""
    sample_id: str
    qa_id: str # Unique session identifier
    question: Any # Prompt or Context
    pred: str # Raw LLM Output
    gold: str # Parsed Tool Calls
    category: str # Category of the question
    evidence: List[str] = field(default_factory=list) # IDs of turns that were the source for this step
    traces: List[Dict] = field(default_factory=list)

class TrainingDataCollector:
    def __init__(self, log_dir: str = "./traj"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        # 分别存储 SFT 原始对数据和 RL 轨迹数据
        self.mem_traj_file = os.path.join(log_dir, "mem_trajectories.jsonl")
        self.qa_traj_file = os.path.join(log_dir, "qa_trajectories.jsonl")
        self.lock = threading.Lock()

    # def log_sft_pair(self, prompt: str, response: str, source: str = ""):
    #     """记录用于 SFT 的 Prompt-Response 对"""
    #     entry = {
    #         "source": source,
    #         "prompt": prompt,
    #         "response": response,
    #         "timestamp": time.time()
    #     }
    #     with self.lock:
    #         with open(self.sft_file, "a", encoding="utf-8") as f:
    #             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_mem_step(self, log: MemTrajectoryLog):
        """记录用于 RL/SFT (GRPO/KTO) 的节点数据"""
        entry = {
            "sample_id": log.sample_id,
            "session_id": log.session_id,
            "benchmark_name": log.benchmark_name,
            "phase": log.phase,
            "op_id": log.op_id,
            "input": log.input_context,
            "output_raw": log.llm_response,
            "output_parsed": log.parsed_output,
            "source_turn_ids": log.source_turn_ids,
            "metadata": log.metadata, # Dict 包含 exact_evidence 等
            "timestamp": time.time()
        }
        with self.lock:
            with open(self.mem_traj_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_qa_step(self, log: QATrajectoryLog):
        """记录用于 qa 的节点数据"""
        entry = {
            "sample_id": log.sample_id,
            "qa_id": log.qa_id,
            "question": log.question,
            "pred": log.pred,
            "gold": log.gold,
            "evidence": log.evidence,
            "traces": log.traces,
            "category": log.category,
            "timestamp": time.time()
        }
        with self.lock:
            with open(self.qa_traj_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
# 全局单例，方便调用
_collector = None

def get_collector(log_dir: str = "./training_data"):
    global _collector
    if _collector is None:
        _collector = TrainingDataCollector(log_dir)
    return _collector