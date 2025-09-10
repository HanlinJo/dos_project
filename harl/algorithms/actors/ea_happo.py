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
    def __init__(self, args, obs_space, act_space, critic_obs_space, device=torch.device("cpu")):
        """
        Initialize EA_HAPPO algorithm, inheriting from OnPolicyBase.
        The signature now correctly accepts 'critic_obs_space'.
        """
        # 1. Call the parent OnPolicyBase's __init__ method
        super(EA_HAPPO, self).__init__(args, obs_space, act_space, device)

        # 2. Copy PPO-related parameters from HAPPO
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

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

        # 6. Initialize EA auxiliary networks
        # It's necessary to save critic_obs_space to get state_dim
        self.critic_obs_space = critic_obs_space
        self.state_dim = self.critic_obs_space.shape[0]
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
                # Assuming state is part of the observation
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