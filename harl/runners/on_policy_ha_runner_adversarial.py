"""Runner for on-policy HARL algorithms with an adversarial policy."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunnerAdversarial(OnPolicyBaseRunner):
    """
    Runner for on-policy HA algorithms with a dedicated adversarial agent.
    The adversary's goal is to learn a policy that maximizes collisions.
    """

    from collections import Counter

    def _get_collision_score(
        self, adversary_action, victim_actions, observation
    ) -> float:
        """
        计算一个综合碰撞分数。
        分数等于所有处于“拥堵资源”中的victim智能体的总数。
        一个资源只要被多于一个智能体（包括adversary和victims）选择，即为“拥堵资源”。

        Args:
            adversary_action: 对抗智能体（adversary）的潜在动作（选择的资源）。
            victim_actions: 一个字典，包含所有受害智能体（victim）的预测动作。
                            格式为 {agent_id: action}。
            observation: 当前的环境观察值（此逻辑中未使用，但为保持API一致性而保留）。

        Returns:
            一个浮点数，表示综合碰撞分数。该分数会计算以下两种情况：
            1. 一个或多个victim与adversary选择了相同的资源。
            2. 多个victim之间选择了相同的资源（即使与adversary的选择不同）。

        示例:
        - adversary_action='A', victim_actions={'v1':'A', 'v2':'B'}:
          资源'A'上有2个智能体（拥堵），v1处于拥堵中。返回 1.0。
        - adversary_action='A', victim_actions={'v1':'A', 'v2':'A'}:
          资源'A'上有3个智能体（拥堵），v1和v2都处于拥堵中。返回 2.0。
        - adversary_action='A', victim_actions={'v1':'B', 'v2':'B'}:
          资源'B'上有2个智能体（拥堵），v1和v2都处于拥堵中。返回 2.0。
        - adversary_action='A', victim_actions={'v1':'B', 'v2':'C'}:
          没有任何资源被多于1个智能体选择（无拥堵）。返回 0.0。
        """
        # 1. 将adversary的动作和所有victim的动作合并到一个列表中
        all_actions = [adversary_action] + list(victim_actions.values())

        # 2. 计算每个动作（资源）被选择了多少次
        action_counts = Counter(all_actions)

        # 3. 找出所有被多于一个智能体选择的“拥堵资源”
        congested_resources = {
            resource for resource, count in action_counts.items() if count > 1
        }

        # 如果没有任何资源拥堵，直接返回0
        if not congested_resources:
            return 0.0

        # 4. 计算有多少个victim选择了这些拥堵的资源
        score = 0.0
        for victim_action in victim_actions.values():
            if victim_action in congested_resources:
                score += 1

        return score

    def _simulate_for_best_actions(self, adversary_id, victim_ids):
        """
        Simulates the outcome of the adversary's actions to find the optimal ones
        for causing collisions.
        """
        print("Starting adversarial simulation to find optimal actions...")
        # Get the most up-to-date policies of the victims
        adversarial_agent = self.actor[adversary_id]
        victim_policies = adversarial_agent.victim_policies

        # Prepare batched observations from buffers for all agents
        # These are needed to get victim responses for each state the adversary was in.
        obs_batch = {
            agent_id: self.actor_buffer[agent_id]
            .obs[:-1]
            .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:])
            for agent_id in range(self.num_agents)
        }
        rnn_states_batch = {
            agent_id: self.actor_buffer[agent_id]
            .rnn_states[0:1]
            .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:])
            for agent_id in range(self.num_agents)
        }
        masks_batch = {
            agent_id: self.actor_buffer[agent_id]
            .masks[:-1]
            .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:])
            for agent_id in range(self.num_agents)
        }

        # Get victims' predicted actions for the entire batch of observations
        victim_action_batch = {}
        with torch.no_grad():
            for vic_id, vic_policy in victim_policies.items():
                # Use the 'act' method for inference to get deterministic or stochastic actions
                # as per the policy's evaluation mode.
                actions, _ = vic_policy.act(
                    obs_batch[vic_id],
                    rnn_states_batch[vic_id],
                    masks_batch[vic_id],
                    deterministic=True,  # Use deterministic actions for planning
                )
                victim_action_batch[vic_id] = _t2n(actions)

        # For each state in the adversary's buffer, find the action that maximizes collisions
        target_actions = []
        batch_size = obs_batch[adversary_id].shape[0]
        # Assuming a discrete action space for the adversary to iterate through.
        num_adv_actions = self.actor[adversary_id].action_space.n

        # This loop can be slow for large batches; vectorizing _get_collision_score is recommended.
        for i in range(batch_size):
            current_obs = {
                agent_id: obs[i] for agent_id, obs in obs_batch.items()
            }
            current_victim_actions = {
                vic_id: acts[i] for vic_id, acts in victim_action_batch.items()
            }

            best_action_for_step = -1
            max_score = -float("inf")

            for potential_action in range(num_adv_actions):
                score = self._get_collision_score(
                    potential_action, current_victim_actions, current_obs
                )
                if score > max_score:
                    max_score = score
                    best_action_for_step = potential_action

            target_actions.append(best_action_for_step)

        print("Adversarial simulation finished.")
        return np.array(target_actions)

    def train(self):
        """
        Train the model, applying a special adversarial training logic for one agent
        and standard on-policy updates for the others.
        """
        actor_train_infos = []

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

        # Compute advantages for all agents from the shared critic
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(
                self.critic_buffer.value_preds[:-1]
            )
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # Normalize advantages if using FP state representation
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # Define agent roles.
        # NOTE: This implementation assumes the adversarial agent has agent_id=0.
        # This should be made configurable in a real application.
        adversary_id = 20
        victim_ids = [i for i in range(self.num_agents) if i != adversary_id]

        # Determine agent update order
        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())

        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(factor)

            # Reshape buffer data to form a single batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # Compute action log probs for the actor before the update.
            # This is needed to calculate the 'factor' for subsequent agents.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # Update actor based on its role (adversary or victim)
            if agent_id == adversary_id:
                # 1. Simulate to find the best actions to cause collisions.
                target_actions = self._simulate_for_best_actions(
                    adversary_id, victim_ids
                )
                # 2. Train the adversarial policy to mimic these target actions.
                # Your AdversarialPolicy's train method must be able to handle 'target_actions'.
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],
                    advantages.copy(),  # Advantages might be used to weight the loss
                    "EP" if self.state_type == "EP" else "FP",
                    target_actions=target_actions,
                )
            else:
                # Standard PPO update for victim agents
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )

            # Compute action log probs for the updated policy to update the factor
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                .obs[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                .rnn_states[0:1]
                .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                .masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                .active_masks[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
            )

            # Update the factor for the next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # Update the shared critic
        critic_train_info = self.critic.train(
            self.critic_buffer, self.value_normalizer
        )

        return actor_train_infos, critic_train_info