import ray
import hydra
import torch
import numpy as np
import re
import json
import sys
import os
import math
import random
from collections import Counter

# Add project root to path
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "Tree-GRPO"))

from verl import DataProto
from verl.trainer.ppo.ray_trainer_memory_ts import RayPPOMemoryTrainer
from verl.trainer.ppo.ray_trainer_ts import ResourcePoolManager, Role


def normalize_text(s: str) -> str:
    if s is None: return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"(^|\s)(a|an|the)(\s|$)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
        
def tokens(s: str):
    s = normalize_text(s)
    return s.split() if s else []

def bleu1_score(pred: str, gold: str) -> float:
    gtoks = tokens(gold)
    ptoks = tokens(pred)
    if len(ptoks) == 0: return 0.0
    gcount = Counter(gtoks)
    pcount = Counter(ptoks)
    clipped = sum(min(pcount[t], gcount[t]) for t in pcount)
    precision = clipped / len(ptoks) if ptoks else 0.0
    if ptoks and gtoks:
        bp = 1.0 if len(ptoks) >= len(gtoks) else math.exp(1 - len(gtoks)/len(ptoks))
    else:
        bp = 0.0
    return bp * precision

class RewardManager():
    """The reward manager for Memory Retrieval."""

    def __init__(self, tokenizer, num_examine, structure_format_score=0., final_format_score=0., retrieval_score=0., format_score=0., evidence_alpha=0.5) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        self.evidence_alpha = evidence_alpha

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        all_scores = []
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # Decode full sequence
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            # Ground Truth from RL dataset
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            
            # Extract the actual answer from the <tool_call> finish or <answer> tag
            # The model is trained to output <tool_call> {"name": "finish", "arguments": {"answer": "..."}} </tool_call>
            # However, standard reward functions expect strict formats.
            # We need to adapt the extraction logic.
            
            # Extract logic customized for Memory Tool format
            pred_answer = self._extract_answer(sequences_str)
            
            # Compute Score
            # We pass pred_answer as solution_str, but compute_score_fn usually parses it too.
            # qa_f1_format.compute_score_f1 usually expects "Answer: ..." or similar.
            # Since we extracted the pure answer, we might need to mock the format or use a simpler metric.
            # Actually, compute_score_fn takes 'solution_str' and does regex.
            # We should wrap our prediction in a way it understands or implement a custom score here.
            
            # Let's implement a custom memory score calculation here directly
            score = self._compute_memory_score(pred_answer, ground_truth)
            
            reward_tensor[i, valid_response_length - 1] = score
            all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"--- Sample {i} ---")
                print(f"Generated: {sequences_str}") # Print last part
                print(f"Pred Answer: {pred_answer}")
                print(f"Ground Truth: {ground_truth}")
                print(f"Score: {score}")

        return all_scores, reward_tensor

    def _extract_answer(self, text: str) -> str:
        # 1. Try finding <tool_call> finish
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text, re.DOTALL)
        for match in reversed(matches): # Look from end
            try:
                tool_data = json.loads(match.strip())
                if tool_data.get("name") == "finish":
                    return str(tool_data.get("arguments", {}).get("answer", ""))
            except:
                continue
                
        # 2. Fallback to <answer> tags (if model degrades)
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        return ""

    def _compute_memory_score(self, pred: str, ground_truth: dict) -> float:
        target = ground_truth.get('target', '')
        if not target:
            return 0.0
            
        # Normalize
        pred = self._normalize_answer(pred)
        target = self._normalize_answer(target)
        
        # Exact Match
        if pred == target:
            return 1.0
            
        # F1 Overlap (Bag of words)
        pred_toks = pred.split()
        target_toks = target.split()
        common = set(pred_toks) & set(target_toks)
        num_same = len(common)
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(target_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        bleu1 = bleu1_score(pred, target)
        return f1 * 0.5 + bleu1 * 0.5

    def _normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
            
        import string
        return white_space_fix(remove_articles(remove_punc(lower(s))))


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        print("Ray is not initialized! run ray.init()...")
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    print("Ray already initialized. Get remote ...")
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    seed = int(config.trainer.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'Exp-B reproducibility seed: {seed}')
    from verl.utils.fs import copy_local_path_from_hdfs
    
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=5, 
                              structure_format_score=config.reward_model.structure_format_score, 
                              final_format_score=config.reward_model.final_format_score,
                              retrieval_score=config.reward_model.retrieval_score,
                              format_score=config.reward_model.format_score,
                              evidence_alpha=config.reward_model.evidence_alpha)

    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOMemoryTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    print(f"=================init_workers================")
    trainer.init_workers()
    print(f"=================fit================")
    trainer.fit()


if __name__ == '__main__':
    main()
