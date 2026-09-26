# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import torch
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
import sys
# Add project root to path to find search_r1
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "Tree-GRPO"))

from search_r1.llm_agent.generation_memory_ts import MemoryGenerationTreeSearchManager
from search_r1.llm_agent.generation_ts import GenerationTreeSearchConfig
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

from verl.trainer.ppo.ray_trainer_ts import (
    Role, ResourcePoolManager, WorkerType, 
    _timer, compute_advantage, compute_data_metrics, compute_timing_metrics, 
    reduce_metrics, apply_kl_penalty, compute_memory_tree_dual_advantage
)

class RayPPOMemoryTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None,
                 debug=False,
                 ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        
        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        ##### for test
        if not debug:
            self._create_dataloader()
            self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=False,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        
        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        reward_tensor_lst = []
        data_source_lst = []
        print("Start validation")
        # Use memory-specific GenerationTreeSearchConfig
        gen_config = GenerationTreeSearchConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            m = 1,
            n = 0,
            l = 0,
            k = 1,
            reward_mode = self.config.actor_rollout_ref.rollout.reward_mode,
            expand_mode = self.config.actor_rollout_ref.rollout.expand_mode,
            evidence_alpha = self.config.reward_model.get('evidence_alpha', 0.5),
        )

        # Use MemoryGenerationTreeSearchManager
        generation_manager = MemoryGenerationTreeSearchManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            reward_fn=self.val_reward_fn, # Use validation reward function
            config=gen_config,
            is_validation = True,
        )

        for batch_dict in self.val_dataloader:
            timing_raw = {}
            batch: DataProto = DataProto.from_single_dict(batch_dict)
            
            # Repeat to match tree search dimensions if needed
            batch = batch.repeat(repeat_times=gen_config.m, interleave=True)
            batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()

            gen_batch = batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['uid', 'data_source', 'reward_model', 'question', 'answer', 'sample_id']
            )
            gen_batch.batch['prompts'] = gen_batch.batch['input_ids']
            
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            
            with _timer('step', timing_raw):
                first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                with _timer('gen', timing_raw):
                    generation_manager.timing_raw = timing_raw
                    # Use Tree Search Logic even for validation to ensure consistency
                    final_gen_batch_output = generation_manager.run_llm_loop_tree_search(
                        gen_batch=gen_batch,
                        initial_input_ids=first_input_ids,
                    )
                
                # Reconstruct batch
                # Tree search output is already expanded. 
                # We need to compute rewards for metric tracking.
                
                # final_gen_batch_output has 'token_level_scores'
                # We can aggregate scores.
                
                reward_tensor = final_gen_batch_output.batch['token_level_scores'].sum(-1).cpu()
                reward_tensor_lst.append(reward_tensor)
                
                # Data source tracking
                # gen_batch has data_source (popped from batch)
                ds = gen_batch.non_tensor_batch['data_source']
                # Expand ds to match reward_tensor shape
                # reward_tensor shape: [Batch * M * K]
                # ds shape: [Batch * M] (already repeated by m)
                # We need to repeat ds by k
                expanded_ds = np.repeat(ds, gen_config.k)
                data_source_lst.append(expanded_ds)

        if not reward_tensor_lst:
            return {}

        reward_tensor = torch.cat(reward_tensor_lst, dim=0)
        data_sources = np.concatenate(data_source_lst, axis=0)
        
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        print(f"Validation metrics: {metric_dict}")
        print("End validation")
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True  
        elif self.config.algorithm.adv_estimator in ['grpo', 'tree', 'tree_inner', 'no_estimator', 'tree_per_token', 'tree_2norm', 'rloo', 'memory_tree']:
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        pprint(f'Check config changes, kl_fix={self.config.algorithm.kl_fix}, use_kl_in_reward={self.config.algorithm.use_kl_in_reward}')

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationTreeSearchConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
            m = self.config.actor_rollout_ref.rollout.ts_m,
            n = self.config.actor_rollout_ref.rollout.ts_n,
            l = self.config.actor_rollout_ref.rollout.ts_l,
            k = self.config.actor_rollout_ref.rollout.ts_k,
            reward_mode = self.config.actor_rollout_ref.rollout.reward_mode,
            expand_mode = self.config.actor_rollout_ref.rollout.expand_mode,
            evidence_alpha = self.config.reward_model.get('evidence_alpha', 0.5),
        )

        # Use MemoryGenerationTreeSearchManager
        generation_manager = MemoryGenerationTreeSearchManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            reward_fn=self.reward_fn,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'[########]epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                ## Repeat for tree search, [BS, ] -> [BS*m, ]
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.ts_m, interleave=True)
                batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                # pop those keys for generation
                gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['uid', 'data_source', 'reward_model', 'question', 'answer', 'sample_id'],
                    )
                gen_batch.batch['prompts'] = gen_batch.batch['input_ids']

                # with _timer('step', timing_raw):
                if True:
                    first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop_tree_search(
                            gen_batch=gen_batch,
                            initial_input_ids=first_input_ids,
                        )

                    for key in final_gen_batch_output.batch.keys():
                        if key != 'token_level_scores':
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                    with torch.no_grad():
                        output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                        final_gen_batch_output = final_gen_batch_output.union(output)
                                        
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.ts_k, interleave=True)
                    batch = batch.union(final_gen_batch_output)

                # balance the number of valid tokens on each dp rank.
                self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                for key in batch.batch.keys():
                    if key != 'old_log_probs' and key != 'token_level_scores':
                        batch.batch[key] = batch.batch[key].long()

                if self.use_reference_policy:
                    # compute reference log_prob
                    with _timer('ref', timing_raw):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with _timer('values', timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with _timer('adv', timing_raw):
                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                    # compute advantages, executed on the driver process
                    batch = compute_advantage(batch,
                                              adv_estimator=self.config.algorithm.adv_estimator,
                                              gamma=self.config.algorithm.gamma,
                                              lam=self.config.algorithm.lam,
                                              )

                # update critic
                if self.use_critic:
                    with _timer('update_critic', timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with _timer('update_actor', timing_raw):
                        if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                            batch, metrics = self._create_loss_mask(batch, metrics)
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                    metrics.update(actor_output_metrics)

                # validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                    self.global_steps % self.config.trainer.test_freq == 0:
                    with _timer('testing', timing_raw):
                        val_metrics: dict = self._validate()
                    metrics.update(val_metrics)

                if (self.config.trainer.save_freq > 0 and \
                        self.global_steps % self.config.trainer.save_freq == 0):
                    # The Exp-B screening launcher sets save_freq=0: metrics and
                    # trajectories are the artifact, while 8-rank checkpoints
                    # would exhaust the shared experiment volume.
                    if self.config.trainer.save_freq > 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, reward_mode=self.config.actor_rollout_ref.rollout.reward_mode))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                print(self.global_steps, metrics)
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)

                    # The screening runner intentionally disables checkpoints to
                    # keep the shared volume available for the complete matrix.
                    if self.config.trainer.save_freq > 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics
