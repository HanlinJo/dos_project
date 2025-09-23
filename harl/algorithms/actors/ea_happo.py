# """EAHAPPO algorithm."""
# import numpy as np
# import torch
# import torch.nn as nn
# from harl.utils.envs_tools import check
# from harl.utils.models_tools import get_grad_norm
# from harl.algorithms.actors.on_policy_base import OnPolicyBase
# # Assuming the following import path for the EA evolver
# from harl.algorithms.ea.mod_neuro_evo import SSNE
# import copy


# class EAHAPPO(OnPolicyBase):
#     def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
#         """Initialize EAHAPPO algorithm.
#         Args:
#             args: (dict) arguments that include EA parameters.
#             obs_space: (gym.spaces or list) observation space.
#             act_space: (gym.spaces) action space.
#             device: (torch.device) device to use for tensor operations.
#         """
#         super(EAHAPPO, self).__init__(args, obs_space, act_space, device)

#         # HAPPO specific parameters
#         self.clip_param = args["clip_param"]
#         self.ppo_epoch = args["ppo_epoch"]
#         self.actor_num_mini_batch = args["actor_num_mini_batch"]
#         self.entropy_coef = args["entropy_coef"]
#         self.use_max_grad_norm = args["use_max_grad_norm"]
#         self.max_grad_norm = args["max_grad_norm"]

#         # EA specific parameters
#         self.pop_size = args.get("pop_size", 5)
#         self.elite_fraction = args.get("elite_fraction", 0.2)
#         self.mutation_prob = args.get("mutation_prob", 0.9)
#         self.prob_reset_and_sup = args.get("prob_reset_and_sup", 0.05)
#         self.frac = args.get("frac", 1.0)
#         self.ea_freq = args.get("ea_freq", 1)
#         self.ea_alpha = args.get("ea_alpha", 1.0) # Distillation strength

#         # Initialize EA population
#         self.population = [copy.deepcopy(self.actor) for _ in range(self.pop_size)]
#         self.evolver = SSNE(args)
#         self.train_count = 0

#     def update(self, sample, main_agent=True, actor_to_update=None):
#         """Update actor network for either the main agent or an EA agent.
#         Args:
#             sample: (Tuple) contains data batch with which to update networks.
#             main_agent: (bool) whether to update the main HAPPO agent.
#             actor_to_update: (StochasticPolicy) the actor to update if not the main agent.
#         Returns:
#             policy_loss: (torch.Tensor) actor(policy) loss value.
#             dist_entropy: (torch.Tensor) action entropies.
#             actor_grad_norm: (torch.Tensor) gradient norm from actor update.
#             imp_weights: (torch.Tensor) importance sampling weights.
#         """
#         (
#             obs_batch,
#             rnn_states_batch,
#             actions_batch,
#             masks_batch,
#             active_masks_batch,
#             old_action_log_probs_batch,
#             adv_targ,
#             available_actions_batch,
#             factor_batch,
#         ) = sample

#         old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
#         adv_targ = check(adv_targ).to(**self.tpdv)
#         active_masks_batch = check(active_masks_batch).to(**self.tpdv)
#         factor_batch = check(factor_batch).to(**self.tpdv)

#         actor = self.actor if main_agent else actor_to_update

#         action_log_probs, dist_entropy, _ = actor.evaluate_actions(
#             obs_batch,
#             rnn_states_batch,
#             actions_batch,
#             masks_batch,
#             available_actions_batch,
#             active_masks_batch,
#         )

#         imp_weights = getattr(torch, self.action_aggregation)(
#             torch.exp(action_log_probs - old_action_log_probs_batch),
#             dim=-1,
#             keepdim=True,
#         )
#         surr1 = imp_weights * adv_targ
#         surr2 = (
#             torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
#             * adv_targ
#         )

#         if self.use_policy_active_masks:
#             policy_action_loss = (
#                 -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
#                 * active_masks_batch
#             ).sum() / active_masks_batch.sum()
#         else:
#             policy_action_loss = -torch.sum(
#                 factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
#             ).mean()

#         policy_loss = policy_action_loss

#         optimizer = self.actor_optimizer if main_agent else torch.optim.Adam(
#             actor.parameters(),
#             lr=self.lr,
#             eps=self.opti_eps,
#             weight_decay=self.weight_decay,
#         )
#         optimizer.zero_grad()
#         (policy_loss - dist_entropy * self.entropy_coef).backward()

#         if self.use_max_grad_norm:
#             actor_grad_norm = nn.utils.clip_grad_norm_(
#                 actor.parameters(), self.max_grad_norm
#             )
#         else:
#             actor_grad_norm = get_grad_norm(actor.parameters())

#         optimizer.step()

#         return policy_loss, dist_entropy, actor_grad_norm, imp_weights

#     def train(self, actor_buffer, advantages, state_type):
#         """Perform a training update using minibatch GD for HAPPO and EA.
#         Args:
#             actor_buffer: (OnPolicyActorBuffer) buffer containing training data.
#             advantages: (np.ndarray) advantages.
#             state_type: (str) type of state.
#         Returns:
#             train_info: (dict) contains training update information.
#         """
#         self.train_count += 1
#         train_info = {}

#         # HAPPO Training for the main agent
#         happo_train_info = self._train_happo(actor_buffer, advantages, state_type)
#         train_info.update({f"main_{k}": v for k, v in happo_train_info.items()})

#         # EA Evolution and Training
#         if self.train_count % self.ea_freq == 0:
#             ea_train_info, elite_actor = self._evolve_and_train_ea(actor_buffer, advantages, state_type)
#             train_info.update({f"ea_{k}": v for k, v in ea_train_info.items()})

#             # Knowledge Distillation
#             distillation_loss = self._knowledge_distillation(actor_buffer, elite_actor)
#             train_info["distillation_loss"] = distillation_loss.item()

#             # Update main actor with distillation loss
#             self.actor_optimizer.zero_grad()
#             (self.ea_alpha * distillation_loss).backward()
#             if self.use_max_grad_norm:
#                 nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
#             self.actor_optimizer.step()

#         return train_info

#     def _train_happo(self, actor_buffer, advantages, state_type):
#         """Train the main HAPPO agent."""
#         train_info = {
#             "policy_loss": 0,
#             "dist_entropy": 0,
#             "actor_grad_norm": 0,
#             "ratio": 0,
#         }

#         if np.all(actor_buffer.active_masks[:-1] == 0.0):
#             return train_info

#         if state_type == "EP":
#             advantages_copy = advantages.copy()
#             advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
#             mean_advantages = np.nanmean(advantages_copy)
#             std_advantages = np.nanstd(advantages_copy)
#             advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

#         for _ in range(self.ppo_epoch):
#             data_generator = self._get_data_generator(actor_buffer, advantages)
#             for sample in data_generator:
#                 policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
#                 train_info["policy_loss"] += policy_loss.item()
#                 train_info["dist_entropy"] += dist_entropy.item()
#                 train_info["actor_grad_norm"] += actor_grad_norm
#                 train_info["ratio"] += imp_weights.mean()

#         num_updates = self.ppo_epoch * self.actor_num_mini_batch
#         for k in train_info.keys():
#             train_info[k] /= num_updates

#         return train_info

#     def _evolve_and_train_ea(self, actor_buffer, advantages, state_type):
#         """Evolve the EA population and train them."""
#         fitness = self._evaluate_fitness(actor_buffer,advantages)
#         elite_index = self.evolver.epoch(self.population, fitness, agent_index=0) # Assuming single agent type for simplicity

#         # Train the evolved population
#         ea_train_info = {
#             "policy_loss": 0,
#             "dist_entropy": 0,
#             "actor_grad_norm": 0,
#             "ratio": 0,
#         }
#         for actor in self.population:
#             for _ in range(self.ppo_epoch):
#                 data_generator = self._get_data_generator(actor_buffer, advantages)
#                 for sample in data_generator:
#                     policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
#                         sample, main_agent=False, actor_to_update=actor
#                     )
#                     ea_train_info["policy_loss"] += policy_loss.item()
#                     ea_train_info["dist_entropy"] += dist_entropy.item()
#                     ea_train_info["actor_grad_norm"] += actor_grad_norm
#                     ea_train_info["ratio"] += imp_weights.mean()
        
#         num_updates = self.ppo_epoch * self.actor_num_mini_batch * self.pop_size
#         for k in ea_train_info.keys():
#             ea_train_info[k] /= num_updates
            
#         return ea_train_info, self.population[elite_index]
        
#     # def _evaluate_fitness(self, actor_buffer):
#     #     """Evaluate the fitness of each actor in the population."""
#     #     fitnesses = []
#     #     with torch.no_grad():
#     #         for actor in self.population:
#     #             # Use negative policy loss as fitness
#     #             policy_loss, _, _, _ = self.update(actor_buffer.sample(actor_buffer.num_agents), main_agent=False, actor_to_update=actor)
#     #             fitnesses.append(-policy_loss.item())
#     #     return fitnesses
#     def _evaluate_fitness(self, actor_buffer, advantages):
#         """Evaluate the fitness of each actor in the population."""
#         fitnesses = []
#         # The update method requires gradients, so we can't use torch.no_grad() here.
#         # Instead, we will detach tensors that don't need gradients inside the update method
#         # for the fitness evaluation pass if needed, but for simplicity, we'll allow grad computation.

#         # We use a simplified fitness metric here if advantages are not ideal,
#         # but since the update method calculates loss based on advantages, let's use that.
#         sample = actor_buffer.sample(advantages)

#         for actor in self.population:
#             # Use negative policy loss as fitness. We need to create a temporary optimizer for this.
#             temp_optimizer = torch.optim.Adam(actor.parameters())

#             # We don't want to actually update the actor weights here, just calculate the loss.
#             # So we perform a forward and backward pass without an optimizer step.
#             policy_loss, _, _, _ = self.update(sample, main_agent=False, actor_to_update=actor)
#             fitnesses.append(-policy_loss.item())

#             # Reset gradients to avoid interference
#             temp_optimizer.zero_grad()

#         return fitnesses

#     def _knowledge_distillation(self, actor_buffer, elite_actor):
#         """Distill knowledge from the elite actor to the main actor."""
#         obs_batch, rnn_states_batch, _, masks_batch, available_actions_batch, active_masks_batch, _, _, _ = actor_buffer.get_whole_data()

#         # Get the policy distribution from the elite actor (teacher)
#         with torch.no_grad():
#             _, _, teacher_dist = elite_actor.evaluate_actions(
#                 obs_batch,
#                 rnn_states_batch,
#                 None, # action is not needed for getting the distribution
#                 masks_batch,
#                 available_actions_batch,
#                 active_masks_batch
#             )

#         # Get the policy distribution from the main actor (student)
#         _, _, student_dist = self.actor.evaluate_actions(
#             obs_batch,
#             rnn_states_batch,
#             None, # action is not needed for getting the distribution
#             masks_batch,
#             available_actions_batch,
#             active_masks_batch
#         )

#         # Calculate KL divergence as the distillation loss
#         distillation_loss = torch.distributions.kl.kl_divergence(teacher_dist, student_dist).mean()
#         return distillation_loss

#     def _get_data_generator(self, actor_buffer, advantages):
#         """Get a data generator for training."""
#         if self.use_recurrent_policy:
#             return actor_buffer.recurrent_generator_actor(
#                 advantages, self.actor_num_mini_batch, self.data_chunk_length
#             )
#         elif self.use_naive_recurrent_policy:
#             return actor_buffer.naive_recurrent_generator_actor(
#                 advantages, self.actor_num_mini_batch
#             )
#         else:
#             return actor_buffer.feed_forward_generator_actor(
#                 advantages, self.actor_num_mini_batch
#             )

"""EAHAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
# Assuming the following import path for the EA evolver
from harl.algorithms.ea.mod_neuro_evo import SSNE
import copy
import gym


# Define the Gene wrapper class to solve the attribute error
class Gene:
    """A wrapper for the actor network to be compatible with the SSNE evolver."""
    def __init__(self, actor_network):
        self.agent_W = torch.nn.ModuleList([actor_network])


class EAHAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize EAHAPPO algorithm.
        Args:
            args: (dict) arguments that include EA parameters.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(EAHAPPO, self).__init__(args, obs_space, act_space, device)

        # HAPPO specific parameters
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        # EA specific parameters
        self.pop_size = args.get("pop_size", 5)
        self.elite_fraction = args.get("elite_fraction", 0.2)
        self.mutation_prob = args.get("mutation_prob", 0.9)
        self.prob_reset_and_sup = args.get("prob_reset_and_sup", 0.05)
        self.frac = args.get("frac", 1.0)
        self.ea_freq = args.get("ea_freq", 1)
        self.ea_alpha = args.get("ea_alpha", 1.0) # Distillation strength

        # Initialize EA population using the Gene wrapper
        self.population = [Gene(copy.deepcopy(self.actor)) for _ in range(self.pop_size)]
        self.evolver = SSNE(args)
        self.train_count = 0

    def update(self, sample, main_agent=True, actor_to_update=None):
        """Update actor network for either the main agent or an EA agent.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            main_agent: (bool) whether to update the main HAPPO agent.
            actor_to_update: (StochasticPolicy) the actor to update if not the main agent.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
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
        factor_batch = check(factor_batch).to(**self.tpdv)

        actor = self.actor if main_agent else actor_to_update

        action_log_probs, dist_entropy, _ = actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        optimizer = self.actor_optimizer if main_agent else torch.optim.Adam(
            actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(actor.parameters())

        optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD for HAPPO and EA.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains training update information.
        """
        self.train_count += 1
        train_info = {}

        # HAPPO Training for the main agent
        happo_train_info = self._train_happo(actor_buffer, advantages, state_type)
        train_info.update({f"main_{k}": v for k, v in happo_train_info.items()})

        # EA Evolution and Training
        if self.train_count % self.ea_freq == 0:
            ea_train_info, elite_actor = self._evolve_and_train_ea(actor_buffer, advantages, state_type)
            train_info.update({f"ea_{k}": v for k, v in ea_train_info.items()})

            # Knowledge Distillation
            distillation_loss = self._knowledge_distillation(actor_buffer, elite_actor)
            train_info["distillation_loss"] = distillation_loss.item()

            # Update main actor with distillation loss
            self.actor_optimizer.zero_grad()
            (self.ea_alpha * distillation_loss).backward()
            if self.use_max_grad_norm:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        return train_info

    def _train_happo(self, actor_buffer, advantages, state_type):
        """Train the main HAPPO agent."""
        train_info = {
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0,
        }

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            data_generator = self._get_data_generator(actor_buffer, advantages)
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def _evolve_and_train_ea(self, actor_buffer, advantages, state_type):
        """Evolve the EA population and train them."""
        fitness = self._evaluate_fitness(actor_buffer,advantages)
        elite_index = self.evolver.epoch(self.population, fitness, agent_index=0) # Assuming single agent type for simplicity

        # Train the evolved population
        ea_train_info = {
            "policy_loss": 0,
            "dist_entropy": 0,
            "actor_grad_norm": 0,
            "ratio": 0,
        }
        for gene in self.population: # Iterate through Gene wrappers
            actor = gene.agent_W[0] # Get the actual actor network
            for _ in range(self.ppo_epoch):
                data_generator = self._get_data_generator(actor_buffer, advantages)
                for sample in data_generator:
                    policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                        sample, main_agent=False, actor_to_update=actor
                    )
                    ea_train_info["policy_loss"] += policy_loss.item()
                    ea_train_info["dist_entropy"] += dist_entropy.item()
                    ea_train_info["actor_grad_norm"] += actor_grad_norm
                    ea_train_info["ratio"] += imp_weights.mean()
        
        num_updates = self.ppo_epoch * self.actor_num_mini_batch * self.pop_size
        for k in ea_train_info.keys():
            ea_train_info[k] /= num_updates
        
        # Return the elite actor network itself, not the Gene wrapper
        elite_actor_network = self.population[elite_index].agent_W[0]
        return ea_train_info, elite_actor_network

    def _evaluate_fitness(self, actor_buffer, advantages):
        """Evaluate the fitness of each actor in the population."""
        fitnesses = []
        sample = actor_buffer.sample(advantages)

        for gene in self.population: # Iterate through Gene wrappers
            actor = gene.agent_W[0] # Get the actual actor network
            # Use negative policy loss as fitness. We need to create a temporary optimizer for this.
            temp_optimizer = torch.optim.Adam(actor.parameters())

            # We don't want to actually update the actor weights here, just calculate the loss.
            # So we perform a forward and backward pass without an optimizer step.
            policy_loss, _, _, _ = self.update(sample, main_agent=False, actor_to_update=actor)
            fitnesses.append(-policy_loss.item())

            # Reset gradients to avoid interference
            temp_optimizer.zero_grad()

        return fitnesses


    def _knowledge_distillation(self, actor_buffer, elite_actor):
        """Distill knowledge from the elite actor to the main actor."""
        obs_batch = check(actor_buffer.obs[:-1]).to(**self.tpdv)
        rnn_states_batch = check(actor_buffer.rnn_states[:-1]).to(**self.tpdv)
        masks_batch = check(actor_buffer.masks[:-1]).to(**self.tpdv)
        active_masks_batch = check(actor_buffer.active_masks[:-1]).to(**self.tpdv)

        if actor_buffer.available_actions is not None:
            available_actions_batch = check(actor_buffer.available_actions[:-1]).to(**self.tpdv)
        else:
            available_actions_batch = None

        # --- Start of the fix ---
        # 根据您的建议，创建一个占位符动作张量。
        # evaluate_actions 函数需要一个动作输入，但其内容在知识蒸馏中无关紧要。
        # 我们只需要确保它的形状、类型和设备是正确的。

        # 获取批处理的维度 (例如：[序列长度, Agent数量])
        batch_dims = obs_batch.shape[:-1]

        # 根据离散或连续动作空间，确定占位符的形状和类型
        if isinstance(self.act_space, gym.spaces.Discrete):
            # 离散空间：动作形状为 (*batch_dims, 1)，类型为 long
            action_shape = (*batch_dims, 1)
            action_dtype = torch.long
        else: # 默认为连续动作空间 (gym.spaces.Box)
            # 连续空间：动作形状为 (*batch_dims, action_dim)，类型为 float
            action_shape = (*batch_dims, self.act_space.shape[0])
            action_dtype = torch.float32

        dummy_action = torch.zeros(action_shape, dtype=action_dtype, device=self.device)
        # --- End of the fix ---

        # Get the policy distribution from the elite actor (teacher)
        with torch.no_grad():
            # 传入占位符动作张量，而不是 None
            _, _, teacher_dist = elite_actor.evaluate_actions(
                obs_batch,
                rnn_states_batch,
                dummy_action,
                masks_batch,
                available_actions_batch,
                active_masks_batch
            )

        # Get the policy distribution from the main actor (student)
        # 同样传入占位符动作张量
        _, _, student_dist = self.actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            dummy_action,
            masks_batch,
            available_actions_batch,
            active_masks_batch
        )

        # Calculate KL divergence as the distillation loss
        kl_div = torch.distributions.kl.kl_divergence(teacher_dist, student_dist)
        # 从 active_masks_batch 的形状中安全地获取目标维度
        target_shape = active_masks_batch.shape
        
        # 使用 view 方法将 kl_div 调整为与 mask 兼容的形状
        kl_div_reshaped = kl_div.view(*target_shape)
    
        # 现在它们的形状兼容，可以安全地进行乘法和求和
        distillation_loss = (kl_div_reshaped * active_masks_batch).sum() / active_masks_batch.sum()
        # distillation_loss = (kl_div * active_masks_batch).sum() / active_masks_batch.sum()

        return distillation_loss

    def _get_data_generator(self, actor_buffer, advantages):
        """Get a data generator for training."""
        if self.use_recurrent_policy:
            return actor_buffer.recurrent_generator_actor(
                advantages, self.actor_num_mini_batch, self.data_chunk_length
            )
        elif self.use_naive_recurrent_policy:
            return actor_buffer.naive_recurrent_generator_actor(
                advantages, self.actor_num_mini_batch
            )
        else:
            return actor_buffer.feed_forward_generator_actor(
                advantages, self.actor_num_mini_batch
            )