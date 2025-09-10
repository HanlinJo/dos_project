# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import copy
# import numpy as np

# # Directly inherit from OnPolicyBase
# from harl.algorithms.actors.on_policy_base import OnPolicyBase
# from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
# from .ea_mine import MINE
# from harl.models.value_function_models.v_net import VNet
# from harl.utils.envs_tools import check
# from harl.utils.models_tools import get_grad_norm

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import copy
# import numpy as np
# from gym import spaces

# # Directly inherit from OnPolicyBase
# from harl.algorithms.actors.on_policy_base import OnPolicyBase
# from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
# from .ea_mine import MINE
# from harl.models.value_function_models.v_net import VNet
# from harl.utils.envs_tools import check
# from harl.utils.models_tools import get_grad_norm

# class EA_HAPPO(OnPolicyBase):
#     def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
#         """
#         Initialize EA_HAPPO algorithm, inheriting from OnPolicyBase.
#         The signature is now modified to calculate critic_obs_space internally.
#         """
#         # 1. Call the parent OnPolicyBase's __init__ method
#         super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)

#         # --- New Calculation Section ---
#         # Calculate global state dimension (m*n)
#         if isinstance(args["state_shape"], int):
#             global_state_dim = args["state_shape"]
#         else:
#             global_state_dim = np.prod(args["state_shape"])
        
#         # Calculate global action dimension (m*a)
#         global_action_dim = 0
#         for space in self.act_space:
#             if isinstance(space, spaces.Discrete):
#                 global_action_dim += space.n
#             elif isinstance(space, spaces.Box):
#                 global_action_dim += space.shape[0]
#             elif isinstance(space, spaces.Tuple):
#                 for sub_space in space.spaces:
#                     if isinstance(sub_space, spaces.Discrete):
#                         global_action_dim += sub_space.n
#                     elif isinstance(sub_space, spaces.Box):
#                         global_action_dim += sub_space.shape[0]

#         # Calculate critic_obs_space dimension based on the user's formula
#         critic_obs_space_dim = global_state_dim + global_action_dim
#         critic_obs_space_shape = (critic_obs_space_dim,)
#         # --- End New Calculation Section ---

#         # 2. Copy PPO-related parameters from HAPPO
#         self.clip_param = args["clip_param"]
#         self.ppo_epoch = args["ppo_epoch"]
#         self.actor_num_mini_batch = args["actor_num_mini_batch"]
#         self.entropy_coef = args["entropy_coef"]
#         self.use_max_grad_norm = args["use_max_grad_norm"]
#         self.max_grad_norm = args["max_grad_norm"]

#         # 3. Get and set EA-specific parameters
#         self.pop_size = args["pop_size"]
#         self.state_alpha = args["state_alpha"]
#         self.EA_alpha = args["EA_alpha"]
#         self.tau = args["tau"]
        
#         # 4. Build EA Actor population structure
#         self.agent_SR = nn.ModuleList()
#         self.agent_W_populations = nn.ModuleList()

#         for i in range(self.num_agents):
#             sr_net = EA_HA_Agent_SR(args, self.obs_space[i], self.act_space[i], device)
#             self.agent_SR.append(sr_net)
            
#             w_population = []
#             main_w_net = EA_HA_Agent_W(args, self.act_space[i], device)
#             w_population.append(main_w_net)
#             for _ in range(self.pop_size):
#                 w = EA_HA_Agent_W(args, self.act_space[i], device)
#                 w.load_state_dict(main_w_net.state_dict())
#                 w_population.append(w)
#             self.agent_W_populations.append(nn.ModuleList(w_population))
        
#         # 5. Rebuild Actor optimizer
#         actor_params = list(self.agent_SR.parameters())
#         for pop in self.agent_W_populations:
#             actor_params += list(pop[0].parameters())
#         self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

#         # 6. Initialize EA auxiliary networks (Modified)
#         # Create a gym.Space object for critic_obs_space
#         self.critic_obs_space = spaces.Box(-np.inf, np.inf, shape=critic_obs_space_shape)
#         # Use the calculated global state dimension for VNet and MINE
#         self.state_dim = global_state_dim
#         self.pevfa_critic = VNet(args, self.state_dim, self.device)
#         self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
#         self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
#         self.hidden_size = args["hidden_size"]
#         self.mine_nets = nn.ModuleList([
#             MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
#         ])
#         self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)

#     def get_actions(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0, pop_id=0):
#         """
#         Get actions for a single agent from a specific policy in the population.
#         """
#         actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
#         action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
#         actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
        
#         return actions, action_log_probs, new_rnn_states

#     def update(self, sample, agent_id=0):
#         """
#         This method is adapted from HAPPO to compute the PPO loss.
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
#         if factor_batch is not None:
#             factor_batch = check(factor_batch).to(**self.tpdv)

#         # Re-evaluate actions using the main policy (pop_id=0)
#         features, _ = self.agent_SR[agent_id](obs_batch, rnn_states_batch, masks_batch)
#         action_logits = self.agent_W_populations[agent_id][0](features)
#         action_log_probs, dist_entropy = self.actors[agent_id].action_dist.evaluate_actions(action_logits, actions_batch, available_actions_batch, active_masks_batch)
        
#         # PPO actor update
#         imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
#         surr1 = imp_weights * adv_targ
#         surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
#         if self.use_policy_active_masks:
#             policy_action_loss = (
#                 -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
#                 * active_masks_batch
#             ).sum() / active_masks_batch.sum()
#         else:
#             policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

#         return policy_action_loss, dist_entropy, features.view(-1, self.hidden_size)

#     def train(self, actor_buffer, critic_buffer, advantages):
#         """
#         Core training function that combines HAPPO loss, EA guidance, and MINE loss.
#         """
#         train_info = {}
        
#         # 1. Train PeVFA Critic
#         state_batch_full = check(critic_buffer.state).to(**self.tpdv)
#         returns_batch_full = check(critic_buffer.returns).to(**self.tpdv)
        
#         pevfa_values = self.pevfa_critic(state_batch_full)
#         # Simplified MSE Loss for PeVFA critic
#         pevfa_critic_loss = (pevfa_values - returns_batch_full).pow(2).mean()
        
#         self.pevfa_critic_optimizer.zero_grad()
#         pevfa_critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
#         self.pevfa_critic_optimizer.step()
#         train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
        
#         # 2. Normalize advantages
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

#         # 3. PPO + EA + MINE joint optimization loop
#         for _ in range(self.ppo_epoch):
#             data_generator = actor_buffer.feed_forward_generator_actor(advantages, self.actor_num_mini_batch)
#             for sample in data_generator:
#                 policy_loss, dist_entropy, features_flat = self.update(sample)

#                 # Get the state batch for auxiliary losses
#                 # Assuming state is part of the observation
#                 obs_batch_for_state = check(sample[0]).to(**self.tpdv)
#                 state_batch = obs_batch_for_state.reshape(-1, obs_batch_for_state.shape[-1])[:, :self.state_dim]

#                 # Calculate EA guidance loss
#                 with torch.no_grad():
#                     pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
                
#                 returns_batch = check(sample[6]).to(**self.tpdv)
#                 action_log_probs_batch = check(sample[5]).to(**self.tpdv)
#                 adv_ea = returns_batch - pevfa_values_guidance
#                 ea_pg_loss = -(action_log_probs_batch * adv_ea).mean()

#                 # MINE loss
#                 min_len = min(len(state_batch), len(features_flat))
#                 mine_loss = -self.mine_nets[0](features_flat[:min_len], state_batch[:min_len])

#                 # Combine all losses
#                 total_loss = (
#                     policy_loss 
#                     - dist_entropy * self.entropy_coef 
#                     + self.EA_alpha * ea_pg_loss 
#                     + self.state_alpha * mine_loss
#                 )

#                 self.actor_optimizer.zero_grad()
#                 self.mine_optimizer.zero_grad()
#                 total_loss.backward()

#                 nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
#                 for pop in self.agent_W_populations:
#                     nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
#                 nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)

#                 self.actor_optimizer.step()
#                 self.mine_optimizer.step()

#                 # Log training info
#                 train_info.setdefault('policy_loss', []).append(policy_loss.item())
#                 train_info.setdefault('dist_entropy', []).append(dist_entropy.item())
#                 train_info.setdefault('ea_pg_loss', []).append(ea_pg_loss.item())
#                 train_info.setdefault('mine_loss', []).append(mine_loss.item())
        
#         # 4. Soft update Target PeVFA Critic and evolve population
#         self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)
#         self.evolve_population()

#         # Average the losses for logging
#         for k in train_info:
#             if isinstance(train_info[k], list):
#                 train_info[k] = np.mean(train_info[k])

#         return train_info

#     def soft_update(self, target, source, tau):
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#     def evolve_population(self):
#         for agent_id in range(self.num_agents):
#             main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
#             for pop_id in range(1, self.pop_size + 1):
#                 mutated_dict = copy.deepcopy(main_w_net_dict)
#                 for key in mutated_dict:
#                     if mutated_dict[key].is_floating_point():
#                         noise = torch.randn_like(mutated_dict[key]) * 0.01
#                         mutated_dict[key] += noise
#                 self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)
import torch
import torch.nn as nn
from torch.optim import Adam
import copy
import numpy as np

# Directly inherit from OnPolicyBase
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
from .ea_mine import MINE
from harl.models.value_function_models.v_net import VNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm

class EA_HAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        """
        Initialize EA_HAPPO algorithm.
        The signature is updated to accept 'share_obs_space' for global state information,
        which is standard in the HARL framework and necessary for the centralized critic.
        """
        # 1. Call the parent OnPolicyBase's __init__ method
        super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)

        # 2. Copy PPO-related parameters
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.num_agents = 2

        # 3. Get and set EA-specific parameters
        self.pop_size = args["pop_size"]
        self.state_alpha = args["state_alpha"]
        self.EA_alpha = args["EA_alpha"]
        self.tau = args["tau"]
        
        # 4. Build EA Actor population structure
        self.agent_SR = nn.ModuleList()
        self.agent_W_populations = nn.ModuleList()

        for i in range(self.num_agents):
            sr_net = EA_HA_Agent_SR(args, self.obs_space[i], self.act_space[i], device)
            self.agent_SR.append(sr_net)
            
            w_population = []
            main_w_net = EA_HA_Agent_W(args, self.act_space[i], device)
            w_population.append(main_w_net)
            for _ in range(self.pop_size):
                w = EA_HA_Agent_W(args, self.act_space[i], device)
                w.load_state_dict(main_w_net.state_dict())
                w_population.append(w)
            self.agent_W_populations.append(nn.ModuleList(w_population))
        
        # 5. Rebuild Actor optimizer
        actor_params = list(self.agent_SR.parameters())
        for pop in self.agent_W_populations:
            actor_params += list(pop[0].parameters())
        self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

        # 6. Initialize EA auxiliary networks using share_obs_space
        # self.share_obs_space is provided by the runner and contains global state information.
        self.state_dim = share_obs_space.shape[0]
        self.pevfa_critic = VNet(args, self.state_dim, self.device)
        self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
        self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
        self.hidden_size = args["hidden_size"]
        self.mine_nets = nn.ModuleList([
            MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
        ])
        self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)

    def get_actions(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, agent_id=0, pop_id=0):
        """
        Get actions for a single agent from a specific policy in the population.
        """
        actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
        action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
        actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
        
        return actions, action_log_probs, new_rnn_states

    def update(self, sample, agent_id=0):
        """
        This method is adapted from HAPPO to compute the PPO loss.
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
        if factor_batch is not None:
            factor_batch = check(factor_batch).to(**self.tpdv)

        # Re-evaluate actions using the main policy (pop_id=0)
        features, _ = self.agent_SR[agent_id](obs_batch, rnn_states_batch, masks_batch)
        action_logits = self.agent_W_populations[agent_id][0](features)
        action_log_probs, dist_entropy = self.actors[agent_id].action_dist.evaluate_actions(action_logits, actions_batch, available_actions_batch, active_masks_batch)
        
        # PPO actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        return policy_action_loss, dist_entropy, features.view(-1, self.hidden_size)

    def train(self, actor_buffer, critic_buffer, advantages):
        """
        Core training function that combines HAPPO loss, EA guidance, and MINE loss.
        """
        train_info = {}
        
        # 1. Train PeVFA Critic
        state_batch_full = check(critic_buffer.state).to(**self.tpdv)
        returns_batch_full = check(critic_buffer.returns).to(**self.tpdv)
        
        pevfa_values = self.pevfa_critic(state_batch_full)
        # Simplified MSE Loss for PeVFA critic
        pevfa_critic_loss = (pevfa_values - returns_batch_full).pow(2).mean()
        
        self.pevfa_critic_optimizer.zero_grad()
        pevfa_critic_loss.backward()
        nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
        self.pevfa_critic_optimizer.step()
        train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
        
        # 2. Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # 3. PPO + EA + MINE joint optimization loop
        for _ in range(self.ppo_epoch):
            data_generator = actor_buffer.feed_forward_generator_actor(advantages, self.actor_num_mini_batch)
            for sample in data_generator:
                policy_loss, dist_entropy, features_flat = self.update(sample)

                # Get the state batch for auxiliary losses
                # Assuming the first part of the observation is the state
                obs_batch_for_state = check(sample[0]).to(**self.tpdv)
                state_batch = obs_batch_for_state.reshape(-1, obs_batch_for_state.shape[-1])[:, :self.state_dim]

                # Calculate EA guidance loss
                with torch.no_grad():
                    pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
                
                returns_batch = check(sample[6]).to(**self.tpdv)
                action_log_probs_batch = check(sample[5]).to(**self.tpdv)
                adv_ea = returns_batch - pevfa_values_guidance
                ea_pg_loss = -(action_log_probs_batch * adv_ea).mean()

                # MINE loss
                min_len = min(len(state_batch), len(features_flat))
                mine_loss = -self.mine_nets[0](features_flat[:min_len], state_batch[:min_len])

                # Combine all losses
                total_loss = (
                    policy_loss 
                    - dist_entropy * self.entropy_coef 
                    + self.EA_alpha * ea_pg_loss 
                    + self.state_alpha * mine_loss
                )

                self.actor_optimizer.zero_grad()
                self.mine_optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
                for pop in self.agent_W_populations:
                    nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)

                self.actor_optimizer.step()
                self.mine_optimizer.step()

                # Log training info
                train_info.setdefault('policy_loss', []).append(policy_loss.item())
                train_info.setdefault('dist_entropy', []).append(dist_entropy.item())
                train_info.setdefault('ea_pg_loss', []).append(ea_pg_loss.item())
                train_info.setdefault('mine_loss', []).append(mine_loss.item())
        
        # 4. Soft update Target PeVFA Critic and evolve population
        self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)
        self.evolve_population()

        # Average the losses for logging
        for k in train_info:
            if isinstance(train_info[k], list):
                train_info[k] = np.mean(train_info[k])

        return train_info

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def evolve_population(self):
        for agent_id in range(self.num_agents):
            main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
            for pop_id in range(1, self.pop_size + 1):
                mutated_dict = copy.deepcopy(main_w_net_dict)
                for key in mutated_dict:
                    if mutated_dict[key].is_floating_point():
                        noise = torch.randn_like(mutated_dict[key]) * 0.01
                        mutated_dict[key] += noise
                self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)                
# # harl/algorithms/actors/ea_happo.py

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import copy
# import random
# import numpy as np

# from .happo import HAPPO
# from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W  # 确保这些自定义模块已存在
# from .ea_mine import MINE  # 确保这个自定义模块已存在
# from harl.models.value_function_models.v_net import VNet
# from harl.utils.envs_tools import check

# class EA_HAPPO(HAPPO):
#     def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
#         """
#         初始化 EA_HAPPO 算法。
#         继承自 HAPPO，并重写了 Actor 结构和训练流程以融入 EA 思想。
#         """
#         model_args = {
#             "hidden_size": args["hidden_size"],
#             "initial_std": args["initial_std"],
#             "use_recurrent_policy": args["use_recurrent_policy"],
#             "use_naive_recurrent_policy": args["use_naive_recurrent_policy"],
#             "lr": args["lr"],
#             "critic_lr": args["critic_lr"],
#             "opti_eps": args["opti_eps"],
#             "weight_decay": args["weight_decay"],
#             "std_x_coef": args["std_x_coef"],
#             "std_y_coef": args["std_y_coef"],
#         }
        
#         # 首先，调用父类HAPPO的初始化方法
#         super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)

#         # --- 1. 获取EA特定参数 ---
#         self.pop_size = args["pop_size"]
#         self.state_alpha = args["state_alpha"]
#         self.EA_alpha = args["EA_alpha"]
#         self.tau = args["tau"]
#         # --- 2. 重写 Actor 为 EA 的种群结构 ---
#         # Actor被分解为共享表征网络(SR)和特定策略网络(W)
#         self.agent_SR = nn.ModuleList()
#         self.agent_W_populations = nn.ModuleList()

#         for i in range(self.num_agents):
#             # 为每个智能体类型创建共享表征网络 (SR)
#             sr_net = EA_HA_Agent_SR(args, obs_space[i], act_space[i], device)
#             self.agent_SR.append(sr_net)

#             # 为每个智能体类型创建W网络种群
#             w_population = []
#             main_w_net = EA_HA_Agent_W(args, act_space[i], device)
#             w_population.append(main_w_net)  # 主Actor的W网络放在索引0
#             for _ in range(self.pop_size):
#                 w = EA_HA_Agent_W(args, act_space[i], device)
#                 w.load_state_dict(main_w_net.state_dict()) # 从主网络初始化
#                 w_population.append(w)
#             self.agent_W_populations.append(nn.ModuleList(w_population))
        
#         # --- 3. 重建 Actor 优化器 ---
#         # 优化器只训练主Actor (SR + W_main)
#         actor_params = list(self.agent_SR.parameters())
#         for pop in self.agent_W_populations:
#             actor_params += list(pop[0].parameters())
#         self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

#         # --- 4. 初始化 EA 辅助网络 ---
#         # a. PeVFA Critic (参数演化价值函数) - 用于评估种群成员
#         # 假设状态维度可以从参数中获取
#         self.state_dim = critic_obs_space.shape[0] if hasattr(critic_obs_space, 'shape') else critic_obs_space[0]
#         self.pevfa_critic = VNet(self.args, self.state_dim, self.device)
#         self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
#         self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
#         # b. MINE (互信息神经估计) - 用于最大化个体状态和全局状态的互信息
#         self.hidden_size = args["hidden_size"]
#         self.mine_nets = nn.ModuleList([
#             MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
#         ])
#         self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)
    
#     def get_actions(self, obs, state, rnn_states_actor, masks, agent_id, pop_id=0, available_actions=None, deterministic=False):
#         """
#         修改 get_actions 以支持从种群中选择策略。
#         Args:
#             pop_id (int): 0表示主Actor, >0 表示种群成员。
#         """
#         # 从SR网络获取特征
#         actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
#         # 从选定的W网络获取动作logit
#         action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
#         # 使用父类的动作分布计算器来采样动作
#         actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
        
#         return actions, action_log_probs, new_rnn_states

#     def train(self, actor_buffer, critic_buffer):
#         """
#         核心训练函数，融合了 HAPPO 损失、EA 指导和 MINE 损失。
#         """
#         train_info = {}
        
#         # --- 1. 训练 PeVFA Critic ---
#         # 这个 Critic 用于评估种群中个体的质量，为EA提供指导信号
        
#         # 从 Critic Buffer 中获取数据
#         state_batch = check(critic_buffer.state).to(**self.tpdv)
#         returns_batch = check(critic_buffer.returns).to(**self.tpdv)
#         active_masks_batch = check(critic_buffer.active_masks).to(**self.tpdv)

#         # PeVFA Critic 的价值预测
#         pevfa_values = self.pevfa_critic(state_batch)
        
#         # 计算 PeVFA Critic 损失 (使用 TD-error)
#         pevfa_critic_loss = self.cal_value_loss(pevfa_values, returns_batch, active_masks_batch)
        
#         self.pevfa_critic_optimizer.zero_grad()
#         pevfa_critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
#         self.pevfa_critic_optimizer.step()

#         # --- 2. 标准 HAPPO Actor 和 Critic 训练 ---
#         # 调用父类的训练方法，完成标准的PPO更新
#         happo_train_info = super().train(actor_buffer, critic_buffer)
#         train_info.update(happo_train_info)
        
#         # --- 3. 计算 EA 指导损失和 MINE 损失 ---
#         ea_pg_loss = 0
#         mine_loss = 0

#         for agent_id in range(self.num_agents):
#             # a) EA Guidance Loss (演化策略梯度)
#             # 这个损失函数的目标是让主Actor的表现优于种群的平均水平
#             # 我们用PeVFA Critic来评估主Actor的动作价值
#             with torch.no_grad():
#                 pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
            
#             # 使用PeVFA的价值作为基线，计算优势函数
#             advantages = returns_batch - pevfa_values_guidance
            
#             action_log_probs = check(actor_buffer[agent_id].action_log_probs).to(**self.tpdv)
            
#             ea_pg_loss_agent = -(action_log_probs * advantages).mean()
#             ea_pg_loss += ea_pg_loss_agent
            
#             # b) MINE Loss (互信息最大化)
#             # 最大化智能体的局部观察编码(features)与全局状态(state)之间的互信息
#             obs_batch = check(actor_buffer[agent_id].obs).to(**self.tpdv)
#             rnn_states_batch = check(actor_buffer[agent_id].rnn_states).to(**self.tpdv)
#             masks_batch = check(actor_buffer[agent_id].masks).to(**self.tpdv)

#             # 重新计算特征
#             features, _ = self.agent_SR[agent_id](obs_batch, rnn_states_batch, masks_batch)

#             # 调整tensor形状以匹配MINE网络的输入
#             state_flat = state_batch.view(-1, self.state_dim)
#             features_flat = features.view(-1, self.hidden_size)
            
#             # 确保批次大小一致
#             min_len = min(len(state_flat), len(features_flat))
#             state_flat = state_flat[:min_len]
#             features_flat = features_flat[:min_len]

#             # MINE损失是互信息的负值，所以最小化它等于最大化互信息
#             mine_loss_agent = -self.mine_nets[agent_id](features_flat, state_flat)
#             mine_loss += mine_loss_agent
        
#         # --- 4. 结合辅助损失并进行优化 ---
#         # 注意: HAPPO的损失已经在 super().train() 中反向传播过了
#         # 这里我们只反向传播EA和MINE的辅助损失
        
#         ea_total_loss = self.EA_alpha * ea_pg_loss + self.state_alpha * mine_loss
        
#         self.actor_optimizer.zero_grad()
#         self.mine_optimizer.zero_grad()
        
#         ea_total_loss.backward()
        
#         nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
#         for pop in self.agent_W_populations:
#             nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
#         nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)
        
#         self.actor_optimizer.step()
#         self.mine_optimizer.step()
        
#         # --- 5. 软更新 Target PeVFA Critic ---
#         self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)

#         # --- 6. 种群进化 ---
#         # 在每个训练周期后，进行一次种群的进化操作
#         self.evolve_population()

#         train_info['ea_pg_loss'] = ea_pg_loss.item()
#         train_info['mine_loss'] = mine_loss.item()
#         train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
        
#         return train_info

#     def soft_update(self, target, source, tau):
#         """软更新目标网络的参数"""
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#     def evolve_population(self):
#         """
#         实现种群的进化操作。
#         这里我们使用一个简单的变异策略：用高斯噪声扰动W网络的权重。
#         """
#         for agent_id in range(self.num_agents):
#             main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
#             for pop_id in range(1, self.pop_size + 1):
#                 mutated_dict = copy.deepcopy(main_w_net_dict)
#                 for key in mutated_dict:
#                     if mutated_dict[key].is_floating_point():
#                         # 添加高斯噪声进行变异
#                         noise = torch.randn_like(mutated_dict[key]) * 0.01 # 变异强度
#                         mutated_dict[key] += noise
#                 self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)

# # harl/algorithms/actors/ea_happo.py

# # import torch
# # import torch.nn as nn
# # from torch.optim import Adam
# # import copy
# # import random
# # import numpy as np

# # from .happo import HAPPO
# # from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
# # from .ea_mine import MINE
# # from harl.models.value_function_models.v_net import VNet
# # from harl.utils.envs_tools import check

# # class EA_HAPPO(HAPPO):
# #     def __init__(self, args, obs_space, act_space, critic_obs_space, device=torch.device("cpu")):
# #         """
# #         初始化 EA_HAPPO 算法。
# #         """
# #         # 首先，调用父类HAPPO的初始化方法
# #         # 注意：父类构造函数需要整个嵌套的 'args' 字典
# #         super(EA_HAPPO, self).__init__(args, obs_space, act_space, critic_obs_space, device)

# #         # --- 1. 获取EA特定参数 (从 'ea' 子字典中读取) ---
# #         ea_args = args["ea"]
# #         self.pop_size = ea_args["pop_size"]
# #         self.state_alpha = ea_args["state_alpha"]
# #         self.EA_alpha = ea_args["EA_alpha"]
# #         self.tau = ea_args["tau"]

# #         # --- 2. 重写 Actor 为 EA 的种群结构 ---
# #         self.agent_SR = nn.ModuleList()
# #         self.agent_W_populations = nn.ModuleList()

# #         for i in range(self.num_agents):
# #             sr_net = EA_HA_Agent_SR(args, self.obs_space[i], self.act_space[i], device)
# #             self.agent_SR.append(sr_net)
            
# #             w_population = []
# #             main_w_net = EA_HA_Agent_W(args, self.act_space[i], device)
# #             w_population.append(main_w_net)
# #             for _ in range(self.pop_size):
# #                 w = EA_HA_Agent_W(args, self.act_space[i], device)
# #                 w.load_state_dict(main_w_net.state_dict())
# #                 w_population.append(w)
# #             self.agent_W_populations.append(nn.ModuleList(w_population))
        
# #         # --- 3. 重建 Actor 优化器 ---
# #         actor_params = list(self.agent_SR.parameters())
# #         for pop in self.agent_W_populations:
# #             actor_params += list(pop[0].parameters())
# #         self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

# #         # --- 4. 初始化 EA 辅助网络 ---
# #         self.state_dim = self.critic_obs_space.shape[0]
# #         self.pevfa_critic = VNet(args, self.state_dim, self.device)
# #         self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
# #         self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
# #         self.hidden_size = args["model"]["hidden_size"]
# #         self.mine_nets = nn.ModuleList([
# #             MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
# #         ])
# #         self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)

# #     # ... (get_actions, train, soft_update, evolve_population 等方法的代码保持不变) ...
# #     def get_actions(self, obs, state, rnn_states_actor, masks, agent_id, pop_id=0, available_actions=None, deterministic=False):
# #         """
# #         修改 get_actions 以支持从种群中选择策略。
# #         Args:
# #             pop_id (int): 0表示主Actor, >0 表示种群成员。
# #         """
# #         # 从SR网络获取特征
# #         actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
# #         # 从选定的W网络获取动作logit
# #         action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
# #         # 使用父类的动作分布计算器来采样动作
# #         actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
        
# #         return actions, action_log_probs, new_rnn_states

# #     def train(self, actor_buffer, critic_buffer):
# #         """
# #         核心训练函数，融合了 HAPPO 损失、EA 指导和 MINE 损失。
# #         """
# #         train_info = {}
        
# #         # --- 1. 训练 PeVFA Critic ---
# #         # 这个 Critic 用于评估种群中个体的质量，为EA提供指导信号
        
# #         # 从 Critic Buffer 中获取数据
# #         state_batch = check(critic_buffer.state).to(**self.tpdv)
# #         returns_batch = check(critic_buffer.returns).to(**self.tpdv)
# #         active_masks_batch = check(critic_buffer.active_masks).to(**self.tpdv)

# #         # PeVFA Critic 的价值预测
# #         pevfa_values = self.pevfa_critic(state_batch)
        
# #         # 计算 PeVFA Critic 损失 (使用 TD-error)
# #         pevfa_critic_loss = self.cal_value_loss(pevfa_values, returns_batch, active_masks_batch)
        
# #         self.pevfa_critic_optimizer.zero_grad()
# #         pevfa_critic_loss.backward()
# #         nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
# #         self.pevfa_critic_optimizer.step()

# #         # --- 2. 标准 HAPPO Actor 和 Critic 训练 ---
# #         # 调用父类的训练方法，完成标准的PPO更新
# #         happo_train_info = super().train(actor_buffer, critic_buffer)
# #         train_info.update(happo_train_info)
        
# #         # --- 3. 计算 EA 指导损失和 MINE 损失 ---
# #         ea_pg_loss = 0
# #         mine_loss = 0

# #         for agent_id in range(self.num_agents):
# #             # a) EA Guidance Loss (演化策略梯度)
# #             # 这个损失函数的目标是让主Actor的表现优于种群的平均水平
# #             # 我们用PeVFA Critic来评估主Actor的动作价值
# #             with torch.no_grad():
# #                 pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
            
# #             # 使用PeVFA的价值作为基线，计算优势函数
# #             advantages = returns_batch - pevfa_values_guidance
            
# #             action_log_probs = check(actor_buffer[agent_id].action_log_probs).to(**self.tpdv)
            
# #             ea_pg_loss_agent = -(action_log_probs * advantages).mean()
# #             ea_pg_loss += ea_pg_loss_agent
            
# #             # b) MINE Loss (互信息最大化)
# #             # 最大化智能体的局部观察编码(features)与全局状态(state)之间的互信息
# #             obs_batch = check(actor_buffer[agent_id].obs).to(**self.tpdv)
# #             rnn_states_batch = check(actor_buffer[agent_id].rnn_states).to(**self.tpdv)
# #             masks_batch = check(actor_buffer[agent_id].masks).to(**self.tpdv)

# #             # 重新计算特征
# #             features, _ = self.agent_SR[agent_id](obs_batch, rnn_states_batch, masks_batch)

# #             # 调整tensor形状以匹配MINE网络的输入
# #             state_flat = state_batch.view(-1, self.state_dim)
# #             features_flat = features.view(-1, self.hidden_size)
            
# #             # 确保批次大小一致
# #             min_len = min(len(state_flat), len(features_flat))
# #             state_flat = state_flat[:min_len]
# #             features_flat = features_flat[:min_len]

# #             # MINE损失是互信息的负值，所以最小化它等于最大化互信息
# #             mine_loss_agent = -self.mine_nets[agent_id](features_flat, state_flat)
# #             mine_loss += mine_loss_agent
        
# #         # --- 4. 结合辅助损失并进行优化 ---
# #         # 注意: HAPPO的损失已经在 super().train() 中反向传播过了
# #         # 这里我们只反向传播EA和MINE的辅助损失
        
# #         ea_total_loss = self.EA_alpha * ea_pg_loss + self.state_alpha * mine_loss
        
# #         self.actor_optimizer.zero_grad()
# #         self.mine_optimizer.zero_grad()
        
# #         ea_total_loss.backward()
        
# #         nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
# #         for pop in self.agent_W_populations:
# #             nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
# #         nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)
        
# #         self.actor_optimizer.step()
# #         self.mine_optimizer.step()
        
# #         # --- 5. 软更新 Target PeVFA Critic ---
# #         self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)

# #         # --- 6. 种群进化 ---
# #         # 在每个训练周期后，进行一次种群的进化操作
# #         self.evolve_population()

# #         train_info['ea_pg_loss'] = ea_pg_loss.item()
# #         train_info['mine_loss'] = mine_loss.item()
# #         train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
        
# #         return train_info

# #     def soft_update(self, target, source, tau):
# #         """软更新目标网络的参数"""
# #         for target_param, param in zip(target.parameters(), source.parameters()):
# #             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

# #     def evolve_population(self):
# #         """
# #         实现种群的进化操作。
# #         这里我们使用一个简单的变异策略：用高斯噪声扰动W网络的权重。
# #         """
# #         for agent_id in range(self.num_agents):
# #             main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
# #             for pop_id in range(1, self.pop_size + 1):
# #                 mutated_dict = copy.deepcopy(main_w_net_dict)
# #                 for key in mutated_dict:
# #                     if mutated_dict[key].is_floating_point():
# #                         # 添加高斯噪声进行变异
# #                         noise = torch.randn_like(mutated_dict[key]) * 0.01 # 变异强度
# #                         mutated_dict[key] += noise
# #                 self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)

# harl/algorithms/actors/ea_happo.py

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import copy
# import random
# import numpy as np

# from .happo import HAPPO
# from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
# from .ea_mine import MINE
# from harl.models.value_function_models.v_net import VNet
# from harl.utils.envs_tools import check

# class EA_HAPPO(HAPPO):
#     def __init__(self, args, obs_space, act_space, critic_obs_space, device=torch.device("cpu")):
#         """
#         Initialize EA_HAPPO algorithm.
#         """
#         # The 'args' dictionary received here is a flattened dictionary of all YAML parameters.
#         # First, we need to reconstruct the nested dictionaries that the parent HAPPO class expects.
#         model_args = {
#             "hidden_size": args["hidden_size"],
#             "initial_std": args["initial_std"],
#             "use_recurrent_policy": args["use_recurrent_policy"],
#             "use_naive_recurrent_policy": args["use_naive_recurrent_policy"],
#             "lr": args["lr"],
#             "critic_lr": args["critic_lr"],
#             "opti_eps": args["opti_eps"],
#             "weight_decay": args["weight_decay"],
#             "std_x_coef": args["std_x_coef"],
#             "std_y_coef": args["std_y_coef"],
#         }
        
#         # We will create a combined dictionary for the parent class.
#         # HAPPO and its base classes expect a flat dictionary, so we pass 'args' directly.
#         super(EA_HAPPO, self).__init__(args, obs_space, act_space, critic_obs_space, device)

#         # --- 1. Get EA-specific parameters (from the top-level flattened 'args') ---
#         self.pop_size = args["pop_size"]
#         self.state_alpha = args["state_alpha"]
#         self.EA_alpha = args["EA_alpha"]
#         self.tau = args["tau"]
        
#         # --- 2. Build EA Actor Population Structure ---
#         self.agent_SR = nn.ModuleList()
#         self.agent_W_populations = nn.ModuleList()

#         for i in range(self.num_agents):
#             sr_net = EA_HA_Agent_SR(model_args, self.obs_space[i], self.act_space[i], device)
#             self.agent_SR.append(sr_net)
            
#             w_population = []
#             main_w_net = EA_HA_Agent_W(model_args, self.act_space[i], device)
#             w_population.append(main_w_net)
#             for _ in range(self.pop_size):
#                 w = EA_HA_Agent_W(model_args, self.act_space[i], device)
#                 w.load_state_dict(main_w_net.state_dict())
#                 w_population.append(w)
#             self.agent_W_populations.append(nn.ModuleList(w_population))
        
#         # --- 3. Rebuild Actor Optimizer for Main Actor ---
#         actor_params = list(self.agent_SR.parameters())
#         for pop in self.agent_W_populations:
#             actor_params += list(pop[0].parameters())
#         self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

#         # --- 4. Initialize EA Auxiliary Networks ---
#         self.state_dim = self.critic_obs_space.shape[0]
#         self.pevfa_critic = VNet(args, self.state_dim, self.device)
#         self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
#         self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
#         self.hidden_size = args["hidden_size"]
#         self.mine_nets = nn.ModuleList([
#             MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
#         ])
#         self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)

#     # ... (The rest of the methods: get_actions, train, soft_update, evolve_population remain the same as the previous correct version) ...
#     def get_actions(self, obs, state, rnn_states_actor, masks, agent_id, pop_id=0, available_actions=None, deterministic=False):
#         actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
#         action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
#         actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
#         return actions, action_log_probs, new_rnn_states

#     def train(self, actor_buffer, critic_buffer):
#         train_info = {}
#         state_batch = check(critic_buffer.state).to(**self.tpdv)
#         returns_batch = check(critic_buffer.returns).to(**self.tpdv)
#         active_masks_batch = check(critic_buffer.active_masks).to(**self.tpdv)
#         pevfa_values = self.pevfa_critic(state_batch)
#         pevfa_critic_loss = self.cal_value_loss(pevfa_values, returns_batch, active_masks_batch)
#         self.pevfa_critic_optimizer.zero_grad()
#         pevfa_critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
#         self.pevfa_critic_optimizer.step()
#         happo_train_info = super().train(actor_buffer, critic_buffer)
#         train_info.update(happo_train_info)
#         ea_pg_loss = 0
#         mine_loss = 0
#         for agent_id in range(self.num_agents):
#             with torch.no_grad():
#                 pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
#             advantages = returns_batch - pevfa_values_guidance
#             action_log_probs = check(actor_buffer[agent_id].action_log_probs).to(**self.tpdv)
#             ea_pg_loss_agent = -(action_log_probs * advantages).mean()
#             ea_pg_loss += ea_pg_loss_agent
#             obs_batch = check(actor_buffer[agent_id].obs).to(**self.tpdv)
#             rnn_states_batch = check(actor_buffer[agent_id].rnn_states).to(**self.tpdv)
#             masks_batch = check(actor_buffer[agent_id].masks).to(**self.tpdv)
#             features, _ = self.agent_SR[agent_id](obs_batch, rnn_states_batch, masks_batch)
#             state_flat = state_batch.view(-1, self.state_dim)
#             features_flat = features.view(-1, self.hidden_size)
#             min_len = min(len(state_flat), len(features_flat))
#             state_flat, features_flat = state_flat[:min_len], features_flat[:min_len]
#             mine_loss_agent = -self.mine_nets[agent_id](features_flat, state_flat)
#             mine_loss += mine_loss_agent
#         ea_total_loss = self.EA_alpha * ea_pg_loss + self.state_alpha * mine_loss
#         self.actor_optimizer.zero_grad()
#         self.mine_optimizer.zero_grad()
#         ea_total_loss.backward()
#         nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
#         for pop in self.agent_W_populations:
#             nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
#         nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)
#         self.actor_optimizer.step()
#         self.mine_optimizer.step()
#         self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)
#         self.evolve_population()
#         train_info['ea_pg_loss'] = ea_pg_loss.item()
#         train_info['mine_loss'] = mine_loss.item()
#         train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
#         return train_info

#     def soft_update(self, target, source, tau):
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#     def evolve_population(self):
#         for agent_id in range(self.num_agents):
#             main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
#             for pop_id in range(1, self.pop_size + 1):
#                 mutated_dict = copy.deepcopy(main_w_net_dict)
#                 for key in mutated_dict:
#                     if mutated_dict[key].is_floating_point():
#                         noise = torch.randn_like(mutated_dict[key]) * 0.01
#                         mutated_dict[key] += noise
#                 self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)

# harl/algorithms/actors/ea_happo.py
# harl/algorithms/actors/ea_happo.py

# import torch
# import torch.nn as nn
# from torch.optim import Adam
# import copy
# import numpy as np

# # 直接继承自 OnPolicyBase
# from .on_policy_base import OnPolicyBase
# from .ea_ha_agent import EA_HA_Agent_SR, EA_HA_Agent_W
# from .ea_mine import MINE
# from harl.models.value_function_models.v_net import VNet
# from harl.utils.envs_tools import check
# from harl.utils.models_tools import get_grad_norm

# class EA_HAPPO(OnPolicyBase):
#     def __init__(self, args, obs_space, act_space, critic_obs_space, device=torch.device("cpu")):
#         """
#         初始化 EA_HAPPO 算法，直接继承自 OnPolicyBase。
#         """
#         # 1. 调用父类 OnPolicyBase 的初始化方法
#         super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)

#         # 2. 从 HAPPO 复制 PPO 相关的参数
#         self.clip_param = args["clip_param"]
#         self.ppo_epoch = args["ppo_epoch"]
#         self.actor_num_mini_batch = args["actor_num_mini_batch"]
#         self.entropy_coef = args["entropy_coef"]
#         self.use_max_grad_norm = args["use_max_grad_norm"]
#         self.max_grad_norm = args["max_grad_norm"]

#         # 3. 获取并设置 EA 特定参数
#         self.pop_size = args["pop_size"]
#         self.state_alpha = args["state_alpha"]
#         self.EA_alpha = args["EA_alpha"]
#         self.tau = args["tau"]
        
#         # 4. 构建 EA Actor 种群结构
#         self.agent_SR = nn.ModuleList()
#         self.agent_W_populations = nn.ModuleList()

#         for i in range(self.num_agents):
#             sr_net = EA_HA_Agent_SR(args, self.obs_space[i], self.act_space[i], device)
#             self.agent_SR.append(sr_net)
            
#             w_population = []
#             main_w_net = EA_HA_Agent_W(args, self.act_space[i], device)
#             w_population.append(main_w_net)
#             for _ in range(self.pop_size):
#                 w = EA_HA_Agent_W(args, self.act_space[i], device)
#                 w.load_state_dict(main_w_net.state_dict())
#                 w_population.append(w)
#             self.agent_W_populations.append(nn.ModuleList(w_population))
        
#         # 5. 重建 Actor 优化器
#         actor_params = list(self.agent_SR.parameters())
#         for pop in self.agent_W_populations:
#             actor_params += list(pop[0].parameters())
#         # 注意：这里的 self.lr 等参数是在 OnPolicyBase 中定义的
#         self.actor_optimizer = Adam(params=actor_params, lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)

#         # 6. 初始化 EA 辅助网络
#         # 必须保存 critic_obs_space 以获取 state_dim
#         self.critic_obs_space = critic_obs_space
#         self.state_dim = self.critic_obs_space.shape[0]
#         self.pevfa_critic = VNet(args, self.state_dim, self.device)
#         self.target_pevfa_critic = copy.deepcopy(self.pevfa_critic)
#         self.pevfa_critic_optimizer = Adam(self.pevfa_critic.parameters(), lr=self.critic_lr, eps=self.opti_eps)
        
#         self.hidden_size = args["hidden_size"]
#         self.mine_nets = nn.ModuleList([
#             MINE(self.hidden_size, self.state_dim) for _ in range(self.num_agents)
#         ])
#         self.mine_optimizer = Adam(self.mine_nets.parameters(), lr=self.critic_lr, eps=self.opti_eps)

#     def get_actions(self, obs, state, rnn_states_actor, masks, agent_id, pop_id=0, available_actions=None, deterministic=False):
#         """
#         修改 get_actions 以支持从种群中选择策略。
#         """
#         actor_features, new_rnn_states = self.agent_SR[agent_id](obs, rnn_states_actor, masks)
#         action_logits = self.agent_W_populations[agent_id][pop_id](actor_features)
        
#         actions, action_log_probs = self.actors[agent_id].action_dist.sample(action_logits, available_actions, deterministic)
        
#         return actions, action_log_probs, new_rnn_states

#     def update(self, sample):
#         """
#         这个方法是从 HAPPO 复制过来的，用于计算 PPO 损失。
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

#         # 重写 `evaluate_actions` 的逻辑，因为 actor 结构变了
#         features, _ = self.agent_SR[0](obs_batch, rnn_states_batch, masks_batch) # 假设agent_id=0
#         action_logits = self.agent_W_populations[0][0](features)
#         action_log_probs, dist_entropy = self.actors[0].action_dist.evaluate_actions(action_logits, actions_batch, available_actions_batch, active_masks_batch)
        
#         # PPO actor update
#         imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
#         surr1 = imp_weights * adv_targ
#         surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        
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

#         return policy_loss, dist_entropy, features.view(-1, self.hidden_size)

#     def train(self, actor_buffer, critic_buffer, advantages):
#         """
#         核心训练函数，融合了 HAPPO 损失、EA 指导和 MINE 损失。
#         """
#         train_info = {}
        
#         # 1. 训练 PeVFA Critic
#         state_batch_full = check(critic_buffer.state).to(**self.tpdv)
#         returns_batch_full = check(critic_buffer.returns).to(**self.tpdv)
#         active_masks_batch_full = check(critic_buffer.active_masks).to(**self.tpdv)

#         pevfa_values = self.pevfa_critic(state_batch_full)
#         # cal_value_loss 方法需要从VCritic类或父类中获取
#         pevfa_critic_loss = (pevfa_values - returns_batch_full).pow(2).mean() # 简化的MSE Loss
        
#         self.pevfa_critic_optimizer.zero_grad()
#         pevfa_critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.pevfa_critic.parameters(), self.max_grad_norm)
#         self.pevfa_critic_optimizer.step()
#         train_info['pevfa_critic_loss'] = pevfa_critic_loss.item()
        
#         # 2. 标准化优势函数
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

#         # 3. PPO + EA + MINE 联合优化循环
#         for _ in range(self.ppo_epoch):
#             data_generator = actor_buffer.feed_forward_generator_actor(advantages, self.actor_num_mini_batch)
#             for sample in data_generator:
#                 policy_loss, dist_entropy, features_flat = self.update(sample)

#                 # 计算 EA 指导损失
#                 state_batch = sample[0][..., :self.state_dim]
#                 with torch.no_grad():
#                     pevfa_values_guidance = self.target_pevfa_critic(state_batch).detach()
                
#                 # adv_targ is not return
#                 returns_batch = sample[6] 
#                 adv_ea = returns_batch - pevfa_values_guidance
#                 action_log_probs = sample[5]
#                 ea_pg_loss = -(action_log_probs * adv_ea).mean()

#                 # MINE 损失
#                 state_flat = state_batch.reshape(-1, self.state_dim)
#                 min_len = min(len(state_flat), len(features_flat))
#                 mine_loss = -self.mine_nets[0](features_flat[:min_len], state_flat[:min_len])

#                 # 结合所有损失
#                 total_loss = (
#                     policy_loss 
#                     - dist_entropy * self.entropy_coef 
#                     + self.EA_alpha * ea_pg_loss 
#                     + self.state_alpha * mine_loss
#                 )

#                 self.actor_optimizer.zero_grad()
#                 self.mine_optimizer.zero_grad()
#                 total_loss.backward()

#                 nn.utils.clip_grad_norm_(self.agent_SR.parameters(), self.max_grad_norm)
#                 for pop in self.agent_W_populations:
#                     nn.utils.clip_grad_norm_(pop[0].parameters(), self.max_grad_norm)
#                 nn.utils.clip_grad_norm_(self.mine_nets.parameters(), self.max_grad_norm)

#                 self.actor_optimizer.step()
#                 self.mine_optimizer.step()

#                 train_info.setdefault('policy_loss', []).append(policy_loss.item())
#                 train_info.setdefault('dist_entropy', []).append(dist_entropy.item())
#                 train_info.setdefault('ea_pg_loss', []).append(ea_pg_loss.item())
#                 train_info.setdefault('mine_loss', []).append(mine_loss.item())
        
#         # 4. 软更新 Target PeVFA Critic 和种群进化
#         self.soft_update(self.target_pevfa_critic, self.pevfa_critic, self.tau)
#         self.evolve_population()

#         for k in train_info:
#             if isinstance(train_info[k], list):
#                 train_info[k] = np.mean(train_info[k])

#         return train_info

#     def soft_update(self, target, source, tau):
#         for target_param, param in zip(target.parameters(), source.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

#     def evolve_population(self):
#         for agent_id in range(self.num_agents):
#             main_w_net_dict = self.agent_W_populations[agent_id][0].state_dict()
#             for pop_id in range(1, self.pop_size + 1):
#                 mutated_dict = copy.deepcopy(main_w_net_dict)
#                 for key in mutated_dict:
#                     if mutated_dict[key].is_floating_point():
#                         noise = torch.randn_like(mutated_dict[key]) * 0.01
#                         mutated_dict[key] += noise
#                 self.agent_W_populations[agent_id][pop_id].load_state_dict(mutated_dict)

# harl/algorithms/actors/ea_happo.py
