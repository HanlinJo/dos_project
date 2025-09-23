# harl/algorithms/actors/ea_happo.py

import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
from harl.common.valuenorm import ValueNorm
from harl.algorithms.ea.mod_neuro_evo import SSNE
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.models.value_function_models.v_net import VNet

# 一个简单的包装类，使actor网络与SSNE模块的接口兼容
class Gene:
    def __init__(self, actor_network):
        self.agent_W = torch.nn.ModuleList([actor_network])

# EAHAPPO 自己的策略类，不再依赖 happo.py
class EAHAPPOPolicy:
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.lr = args["lr"]
        self.obs_space = obs_space
        self.act_space = act_space
        self.num_agents = num_agents

        # HAPPAPolicy 的核心是包含 actor 和 critic
        # 注意：actor是在 EAHAPPO 主类中创建的，这里只是占位符
        # critic 则在这里创建
        self.actor = nn.ModuleList() # 将在 EAHAPPO 中被填充
        self.critic = VNet(args, self.obs_space[0].shape, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def evaluate_actions(self, obs_batch, action_batch, available_actions_batch, active_masks_batch):
        """
        从 happo.py 复制并适配的核心函数，用于PPO更新
        """
        action_log_probs = []
        dist_entropy = []

        for i in range(self.num_agents):
            # 获取单个智能体的actor网络和数据
            actor = self.actor[i]
            obs_batch_agent = obs_batch[:, i]
            action_batch_agent = action_batch[:, i]
            available_actions_batch_agent = available_actions_batch[:, i] if available_actions_batch is not None else None
            active_masks_batch_agent = active_masks_batch[:, i]

            # 评估动作
            _, agent_action_log_probs, agent_dist_entropy = actor.evaluate_actions(
                obs_batch_agent,
                action_batch_agent,
                available_actions_batch_agent,
                active_masks_batch_agent,
            )

            action_log_probs.append(agent_action_log_probs)
            dist_entropy.append(agent_dist_entropy)

        # 堆叠结果
        action_log_probs = torch.stack(action_log_probs, dim=1)
        dist_entropy = torch.stack(dist_entropy, dim=1).mean()

        return action_log_probs, dist_entropy

class EAHAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        super(EAHAPPO, self).__init__(args, obs_space, act_space, device)

        # 复制 HAPPO 的参数初始化
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.num_agents = 2

        # --- EA特定参数设置 ---
        self.pop_size = self.args.get('pop_size', 10)
        self.ea_freq = self.args.get('ea_freq', 10)
        print("EAHAPPO initialized with population size:", self.pop_size)

        ea_args = SimpleNamespace(
            pop_size=self.pop_size,
            frac=self.args.get('frac', 0.02),
            prob_reset_and_sup=self.args.get('prob_reset_and_sup', 0.05),
            elite_fraction=self.args.get('elite_fraction', 0.1),
            mutation_prob=self.args.get('mutation_prob', 0.9)
        )
        self.ea = SSNE(ea_args)
        
        # 创建EA actor种群
        self.actor_population = nn.ModuleList([
            nn.ModuleList([StochasticPolicy(args, obs_space[i], act_space[i], device) for _ in range(self.pop_size)])
            for i in range(self.num_agents)
        ])
        
        # 创建我们自己的策略对象
        self.policy = EAHAPPOPolicy(args, obs_space, act_space, self.num_agents, device)

        # 将EA种群中的主个体与policy对象链接
        for i in range(self.num_agents):
            self.policy.actor.append(self.actor_population[i][0])
        # 为actor_optimizer重新分配参数
        self.policy.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.policy.lr)
        
        self.value_normalizer = ValueNorm(1).to(self.device)
        self.fitnesses = [[0.0] * self.pop_size for _ in range(self.num_agents)]
        
    def train(self, buffer, update_actor=True):
        if not update_actor:
            return {}

        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        advantages = buffer.advantages
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_actor(advantages, self.actor_num_mini_batch)
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        self.train_step += 1

        if self.args.get('use_ea', False) and self.train_step > 0 and self.train_step % self.ea_freq == 0:
            print(f"\n--- Running EA at training step {self.train_step} ---")
            self.evolve_population(buffer)
            print("--- EA Finished ---")

        return train_info
        
    def update(self, sample):
        (obs_batch, _, actions_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch) = sample
        old_action_log_probs_batch = old_action_log_probs_batch.to(self.device)
        adv_targ = adv_tarrg.to(self.device)
        active_masks_batch = active_masks_batch.to(self.device)

        action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch, actions_batch, available_actions_batch, active_masks_batch)

        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        policy_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)).mean()

        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.entropy_coef).backward()
        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def evaluate_population_fitness(self, buffer):
        with torch.no_grad():
            for agent_id in range(self.num_agents):
                for i in range(self.pop_size):
                    actor = self.actor_population[agent_id][i]
                    obs_batch = torch.tensor(buffer.obs[0, :, agent_id]).to(self.device)
                    rnn_states_batch = torch.tensor(buffer.rnn_states[0, :, agent_id]).to(self.device)
                    masks_batch = torch.tensor(buffer.masks[0, :, agent_id]).to(self.device)
                    _, action_log_probs, _ = actor(obs_batch, rnn_states_batch, masks_batch)
                    fitness = action_log_probs.mean().item()
                    self.fitnesses[agent_id][i] = fitness

    def evolve_population(self, buffer):
        self.evaluate_population_fitness(buffer)
        for agent_id in range(self.num_agents):
            gene_population = [Gene(actor) for actor in self.actor_population[agent_id]]
            fitnesses = self.fitnesses[agent_id]
            self.ea.epoch(pop=gene_population, fitness_evals=fitnesses, agent_index=0)
            for i in range(self.pop_size):
                evolved_actor = gene_population[i].agent_W[0]
                self.actor_population[agent_id][i].load_state_dict(evolved_actor.state_dict())
            best_params = self.actor_population[agent_id][0].state_dict()
            self.policy.actor[agent_id].load_state_dict(best_params)