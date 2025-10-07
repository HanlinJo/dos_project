# harl/algorithms/actors/happo_adversarial.py

import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.models.policy_models.adversarial_policy import AdversarialPolicy
from gym import spaces
from harl.models.policy_models.stochastic_policy import StochasticPolicy

class HAPPOAdversarial(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(HAPPOAdversarial, self).__init__(args, obs_space, act_space, device)
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.adversarial_policy = AdversarialPolicy(self.args, obs_space, self.act_space, self.device)
        self.adversarial_optimizer = torch.optim.Adam(
            self.adversarial_policy.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.victim_policies = self._load_victim_models()

    def get_actions(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        # 参数名从 victim_obs 改为 obs
        return self.adversarial_policy(obs, rnn_states, masks, available_actions, deterministic)

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        # 参数名从 victim_obs 改为 obs
        return self.adversarial_policy.evaluate_actions(obs, rnn_states, action, masks, available_actions, active_masks)
    
    def _load_victim_models(self):
        """加载所有预训练的victim模型。"""
        print("Loading all pre-trained victim models for simulation...")
        victim_model_path = self.args["rl_vehicle_model_path"]
        num_vehicles = self.args['num_vehicles']

        # 计算victim模型的观测/动作空间（仅用于实例化模型结构）
        num_slots = self.args['num_slots']
        num_subchannels = self.args['num_subchannels']
        if num_subchannels <= 1: num_subchannel_choices = 1
        else: num_subchannel_choices = num_subchannels - 1
        victim_obs_dim = (num_slots * num_subchannels) + 5 + 1 + num_vehicles
        victim_act_dim = num_slots * num_subchannel_choices
        victim_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(victim_obs_dim,), dtype=np.float32)
        victim_act_space = spaces.Discrete(victim_act_dim)

        loaded_policies = []
        for i in range(num_vehicles):
            # 实例化策略网络结构
            policy = StochasticPolicy(self.args, victim_obs_space, victim_act_space, device=self.device)
            # 加载权重
            model_file = f"{victim_model_path}/actor_agent{i}.pt"
            state_dict = torch.load(model_file, map_location=self.device,weights_only=True)
            policy.load_state_dict(state_dict)
            policy.eval()
            loaded_policies.append(policy)
        
        print(f"Successfully loaded {len(loaded_policies)} victim models.")
        return loaded_policies
    
    def update(self, sample):
        """Update adversarial network."""
        (
            victim_obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            victim_obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_action_loss = -torch.min(surr1, surr2).mean()

        policy_loss = policy_action_loss

        self.adversarial_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.adversarial_policy.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.adversarial_policy.parameters())

        self.adversarial_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
    
        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info
    
        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
    
        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
    
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )
    
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
    
        num_updates = self.ppo_epoch * self.actor_num_mini_batch
    
        for k in train_info.keys():
            train_info[k] /= num_updates
    
        return train_info