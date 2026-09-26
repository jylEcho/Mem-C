import torch
import re
import json
import requests
import uuid
import sys
import os
import time
import numpy as np
from typing import List, Any, Tuple, Dict

# Add project root to path to find trajectory_logger
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "Tree-GRPO"))

from search_r1.llm_agent.generation_ts import LLMGenerationTreeSearchManager
from search_r1.llm_agent.generation import LLMGenerationManager
from search_r1.llm_agent.tree_node import TreeNode, DEBUG, dprint
from verl import DataProto
from trajectory_logger import get_collector, MemTrajectoryLog

class MemoryGenerationTreeSearchManager(LLMGenerationTreeSearchManager):
    def __init__(self, tokenizer, actor_rollout_wg, reward_fn, config, is_validation=False, log_dir="./traj"):
        super().__init__(tokenizer, actor_rollout_wg, reward_fn, config, is_validation)
        self.collector = get_collector(log_dir)

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        processed_str = []
        for resp in responses_str:
            # Fix closing tag if truncated
            if "<tool_call>" in resp:
                if "</tool_call>" not in resp:
                     resp += "</tool_call>"
                else:
                     resp = resp.split("</tool_call>")[0] + "</tool_call>"
            processed_str.append(resp)
            
        responses = self._batch_tokenize(processed_str)
        return responses, processed_str

    def gen_action_chain(self, gen_batch, node_list, turns_stats):
        """
        Overridden to pass node_list to execute_predictions for memory tracking.
        Mostly copied from parent, but calls self.execute_predictions with node_list.
        """
        # ... Init logic ...
        init_responses_list = []
        init_responses_with_info_mask_list = []
        init_turns_mask_list = []
        for i, node in enumerate(node_list):
            if node.responses is not None:
                responses = node.responses
                responses_with_info_mask = node.responses_with_info_mask
                turns_mask = node.turns_mask
            else:
                responses = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                responses_with_info_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
                turns_mask = torch.tensor([], dtype=gen_batch.batch['input_ids'].dtype).to(gen_batch.batch['input_ids'].device)
            init_responses_list.append(responses)
            init_responses_with_info_mask_list.append(responses_with_info_mask)
            init_turns_mask_list.append(turns_mask)
            
        init_responses = self.tensor_fn.pad_and_stack(init_responses_list, pad_to_left=False)
        init_responses_with_info_mask = self.tensor_fn.pad_and_stack(init_responses_with_info_mask_list, pad_to_left=False)
        init_turns_mask = self.tensor_fn.pad_and_stack(init_turns_mask_list, pad_to_left=False)
        original_right_side = {
            'responses': init_responses, 
            'responses_with_info_mask': init_responses_with_info_mask,
            'turns_mask': init_turns_mask
        }

        tmp_node_list = node_list
        initial_input_ids = gen_batch.batch['input_ids'][:, -self.config.max_start_length:]
        active_mask = torch.ones(initial_input_ids.shape[0], dtype=torch.bool)
        valid_action_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(initial_input_ids.shape[0], dtype=torch.int)
        
        # fix bug. inherit parent node stats
        for i, node in enumerate(tmp_node_list):
            valid_action_stats[i] = node.valid_action_stats
            valid_search_stats[i] = node.valid_search_stats

        rollings = gen_batch
        
        for step in range(self.config.max_turns):
            need_act_mask = turns_stats < self.config.max_turns
            need_act_mask = need_act_mask.to(active_mask.dtype) * active_mask

            if not need_act_mask.sum():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            rollings_active = DataProto.from_dict(
                    tensors={k: v[need_act_mask] for k, v in rollings.batch.items()},
                    meta_info=rollings.meta_info,
                )
            gen_output = self._generate_with_gpu_padding(rollings_active)

            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, need_act_mask)
            # print(f"==========responses_str={responses_str}")
            # --- MODIFIED: Pass node_list to execute_predictions ---
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, need_act_mask, node_list=tmp_node_list, gen_batch=gen_batch
            )
            # print(f"==========next_obs={next_obs}")
            # print(f"==========dones={dones}")
            # print(f"==========valid_action={valid_action}")
            # print(f"==========is_search={is_search}")
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            curr_action_mask = need_act_mask
            active_mask = active_mask * curr_active_mask
            turns_stats[need_act_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            original_right_side = self._update_right_side(original_right_side, responses_ids, turns_stats, next_obs_ids)

            # create tree node
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                if not curr_action_mask[i]:
                    continue

                node_uid = str(uuid.uuid4())
                input_ids = rollings.batch['input_ids'][i]
                position_ids = rollings.batch['position_ids'][i]
                attention_mask = rollings.batch['attention_mask'][i]
                responses = original_right_side['responses'][i]
                responses_with_info_mask = original_right_side['responses_with_info_mask'][i]
                turns_mask = original_right_side['turns_mask'][i]

                parent_node = tmp_node_list[i]
                prompts = parent_node.prompts
                tree_uid = parent_node.tree_uid

                new_node = TreeNode(
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    responses=responses,
                    responses_with_info_mask=responses_with_info_mask,
                    turns_mask=turns_mask,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=bool(curr_active_mask[i].item()),
                    valid_action_stats=int(valid_action_stats[i].item()),
                    valid_search_stats=int(valid_search_stats[i].item()),
                    depth=parent_node.depth+1,
                    is_leaf=bool(dones[i]),
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )
                
                if hasattr(parent_node, 'seen_turns'):
                    new_node.seen_turns = parent_node.seen_turns.copy()
                else:
                    new_node.seen_turns = set()
                
                new_node.retrieved_turns = getattr(parent_node, '_last_step_retrieved_turns', set()).copy()
                new_node.format_valid = getattr(parent_node, '_last_step_format_valid', True)
                rm_data = gen_batch.non_tensor_batch.get('reward_model')
                if rm_data is not None:
                    gt = rm_data[i] if hasattr(rm_data, '__getitem__') else {}
                    if isinstance(gt, dict):
                        new_node.evidence_turns = gt.get('ground_truth', {}).get('evidence', [])
                
                parent_node.add_child(new_node)
                tmp_node_list[i] = new_node
            
            
        # final LLM rollout with forced answer
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    meta_info=rollings.meta_info,
                )
            
            # --- Add forced answer prompt ---
            forced_prompt = " You have reached the maximum number of turns. Please answer the question now. Your response must be in the format of <tool_call> {{'name': 'finish', 'arguments': {{'answer': 'answer'}}}} </tool_call>."
            forced_ids = self.tokenizer(forced_prompt, return_tensors='pt', add_special_tokens=False)['input_ids']
            forced_ids = forced_ids.to(rollings_active.batch['input_ids'].device)
            forced_ids_batch = forced_ids.repeat(rollings_active.batch['input_ids'].shape[0], 1)
            
            # Concatenate
            rollings_active.batch['input_ids'] = torch.cat([rollings_active.batch['input_ids'], forced_ids_batch], dim=1)
            
            # Update attention mask and position ids
            new_att_mask = self.tensor_fn.create_attention_mask(rollings_active.batch['input_ids'])
            new_pos_ids = self.tensor_fn.create_position_ids(new_att_mask)
            rollings_active.batch['attention_mask'] = new_att_mask
            rollings_active.batch['position_ids'] = new_pos_ids
            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            if DEBUG:
                rollout_token_cnt += gen_output.batch['responses'].shape[0] * gen_output.batch['responses'].shape[1]

            # Execute in environment and process observations
            # We pass do_search=False to hint (though current implementation ignores it, the forced prompt implies finish)
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, node_list=tmp_node_list, gen_batch=gen_batch
            )

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            curr_action_mask = active_mask.clone()
            turns_stats[active_mask] += 1
            active_mask = active_mask * curr_active_mask
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turns_stats,
                next_obs_ids
            )

            # create tree node
            for i in range(gen_batch.batch['input_ids'].shape[0]):
                if not curr_action_mask[i]:
                    continue

                node_uid = str(uuid.uuid4())
                input_ids = rollings.batch['input_ids'][i]
                position_ids = rollings.batch['position_ids'][i]
                attention_mask = rollings.batch['attention_mask'][i]

                responses = original_right_side['responses'][i]
                responses_with_info_mask = original_right_side['responses_with_info_mask'][i]
                turns_mask = original_right_side['turns_mask'][i]

                # prompts not in rollings
                parent_node = tmp_node_list[i]
                prompts = parent_node.prompts
                tree_uid = parent_node.tree_uid

                new_node = TreeNode(
                    tree_uid=tree_uid,
                    node_uid=node_uid,
                    prompts=prompts,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    responses=responses,
                    responses_with_info_mask=responses_with_info_mask,
                    turns_mask=turns_mask,
                    parent_node=parent_node,
                    is_root=False,
                    is_active=bool(curr_active_mask[i].item()),
                    valid_action_stats=int(valid_action_stats[i].item()),
                    valid_search_stats=int(valid_search_stats[i].item()),
                    depth=parent_node.depth+1,
                    is_leaf=True,
                    reward_mode=self.config.reward_mode,
                    tensor_fn=self.tensor_fn,
                )
                
                if hasattr(parent_node, 'seen_turns'):
                    new_node.seen_turns = parent_node.seen_turns.copy()
                else:
                    new_node.seen_turns = set()

                new_node.retrieved_turns = getattr(parent_node, '_last_step_retrieved_turns', set()).copy()
                new_node.format_valid = getattr(parent_node, '_last_step_format_valid', True)
                rm_data = gen_batch.non_tensor_batch.get('reward_model')
                if rm_data is not None:
                    gt = rm_data[i] if hasattr(rm_data, '__getitem__') else {}
                    if isinstance(gt, dict):
                        new_node.evidence_turns = gt.get('ground_truth', {}).get('evidence', [])
                    
                parent_node.add_child(new_node)
                tmp_node_list[i] = new_node

    def postprocess_predictions(self, predictions: List[str]) -> Tuple[List[str], List[Any]]:
        actions = []
        contents = []
        for prediction in predictions:
            # Look for <tool_call> JSON </tool_call>
            pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(pattern, prediction, re.DOTALL)
            if match:
                content = match.group(1).strip()
                try:
                    tool_data = json.loads(content)
                    tool_name = tool_data.get("name")
                    action = "finish" if tool_name == "finish" else "tool"
                    actions.append(action)
                    contents.append(tool_data)
                except:
                    # Invalid JSON
                    actions.append(None)
                    contents.append(None)
            else:
                actions.append(None)
                contents.append(None)
        return actions, contents

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, node_list=None, gen_batch=None) -> Tuple[List[str], List[int], List[int], List[int]]:
        cur_actions, tool_datas = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        for i, (action, tool_data) in enumerate(zip(cur_actions, tool_datas)):
            node = node_list[i] if node_list else None

            if not active_mask[i]:
                next_obs.append('')
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
                if node is not None:
                    node._last_step_retrieved_turns = set()
                    node._last_step_format_valid = True
                continue
            
            if action is None:
                next_obs.append("\nInvalid format. Use <tool_call> {\"name\": \"...\", \"arguments\": {...}} </tool_call>\n")
                dones.append(0)
                valid_action.append(0)
                is_search.append(0)
                if node is not None:
                    node._last_step_retrieved_turns = set()
                    node._last_step_format_valid = False
            elif action == "finish":
                answer = tool_data.get("arguments", {}).get("answer", "")
                next_obs.append(answer)
                print(f"==========answer={answer}")
                dones.append(1)
                valid_action.append(1)
                is_search.append(0)
                if node is not None:
                    node._last_step_retrieved_turns = set()
                    node._last_step_format_valid = True
            elif action == "tool":
                sample_id = gen_batch.non_tensor_batch['sample_id'][i]
                if node is not None and not hasattr(node, 'seen_turns'):
                    node.seen_turns = set()
                
                try:
                    old_seen = node.seen_turns.copy() if node is not None else set()
                    payload = {
                        "tool_name": tool_data["name"],
                        "arguments": tool_data.get("arguments", {}),
                        "sample_id": sample_id,
                        "seen_turns": list(node.seen_turns) if node is not None else []
                    }
                    response = requests.post(self.config.search_url, json=payload)
                    res_json = response.json()
                    observation = res_json.get("result", "")
                    mem_metadatas = res_json.get("mem_metadatas", [])
                    updated_seen_turns = res_json.get("updated_seen_turns", [])

                    # Extract turn IDs from mem_metadatas
                    retrieved_turn_ids = set()
                    for meta in mem_metadatas:
                        if meta and "source_turn_ids" in meta:
                            source_turn_ids = meta["source_turn_ids"]
                            # source_turn_ids is a list of lists, flatten it
                            if isinstance(source_turn_ids, list):
                                for turn_ids in source_turn_ids:
                                    if isinstance(turn_ids, list):
                                        retrieved_turn_ids.update(turn_ids)
                                    else:
                                        retrieved_turn_ids.add(turn_ids)

                    if node is not None:
                        # seen_turns still tracks content strings for deduplication in API
                        node.seen_turns = set(updated_seen_turns)
                        # but _last_step_retrieved_turns should be turn IDs for reward calculation
                        node._last_step_retrieved_turns = retrieved_turn_ids
                        node._last_step_format_valid = True
                    
                    original_question = gen_batch.non_tensor_batch['question'][i]
                    full_obs = f"The retrieved memories are: <observation>{observation}</observation>\n Verify if you are confident enough to answer the <question>{original_question}</question> correctly and sufficiently based on the above memories; if not, continue using the tools to retrieve more information."
                    next_obs.append(full_obs)
                    
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    
                except Exception as e:
                    next_obs.append(f"\nTool execution error: {e}\n")
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    if node is not None:
                        node._last_step_retrieved_turns = set()
                        node._last_step_format_valid = False

        return next_obs, dones, valid_action, is_search

    def _compute_final_scores(self, root_list, final_node_list, final_output):
        """
        Compute node-wise rewards: R(v) = I_fmt(v) * (alpha * Evid(v) + Perform(v))
        where Perform(leaf) = F1, Perform(internal) = mean(children Perform).
        """
        print('===========start calculate memory node-wise reward')
        original_scores, _ = self.reward_fn(final_output)
        for i, node in enumerate(final_node_list):
            print(f'node:{node.node_uid}, original_score={original_scores[i]}')
            node.set_leaf_original_score(original_scores[i])

        alpha = getattr(self.config, 'evidence_alpha', 0.5)
        for root in root_list:
            TreeNode.compute_perform_from_root(root)
            TreeNode.compute_node_rewards(root, alpha=alpha)

        for root in root_list:
            for node in root.get_subtree_nodes():
                print(f'  node:{node.node_uid}, depth={node.depth}, '
                      f'fmt={node.format_valid}, evid_reward={node.evid_reward:.4f}, '
                      f'perform={node.perform_score:.4f}, R(v)={node.node_reward:.4f}')

    def _post_compute_scores(self, root_list, final_node_list, final_output):
        """Store per-sample node rewards and segment boundaries for advantage computation."""
        node_rewards_list = []
        node_boundaries_list = []

        for leaf_node in final_node_list:
            path_data = []
            n = leaf_node
            while n and not n.is_root:
                valid_len = int(self.tensor_fn.create_attention_mask(n.responses).sum().item())
                path_data.append((valid_len, n.node_reward))
                n = n.parent_node
            path_data.reverse()

            rewards = [r for _, r in path_data]
            boundaries = [0] + [l for l, _ in path_data]

            node_rewards_list.append(rewards)
            node_boundaries_list.append(boundaries)

        final_output.non_tensor_batch['node_rewards'] = np.array(node_rewards_list, dtype=object)
        final_output.non_tensor_batch['node_boundaries'] = np.array(node_boundaries_list, dtype=object)

        trajectory_dir = os.environ.get("EXP_B_TRAJ_DIR")
        if trajectory_dir and self.selection_records:
            os.makedirs(trajectory_dir, exist_ok=True)
            record = {
                "timestamp": time.time(),
                "expand_mode": self.config.expand_mode,
                "coverage_target": self.config.coverage_target,
                "records": self.selection_records,
            }
            with open(os.path.join(trajectory_dir, "selection_metrics.jsonl"), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
