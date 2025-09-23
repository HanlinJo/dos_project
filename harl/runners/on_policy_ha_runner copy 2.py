"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self):
        """Train the model."""
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

        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                :-1
            ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
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

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
            
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor
            
            # Check if the current actor for this agent uses the EA logic
            is_ea_actor = hasattr(self.actor[agent_id], 'use_ea') and self.actor[agent_id].use_ea

            if is_ea_actor:
                # =================================================================================
                # EA-Specific Logic: Prepare data for each individual in the population
                # =================================================================================
                population_buffers = []
                population_advantages = []

                print(f"Runner: Preparing data for EA population of agent {agent_id}...")

                for individual_actor in self.actor[agent_id].population:
                    # This is a simplified approach: we let each individual train on the data
                    # collected by the main actor.
                    individual_buffer = self.actor_buffer[agent_id]
                    population_buffers.append(individual_buffer)

                    # For fitness evaluation, we reuse the globally computed advantages.
                    if self.state_type == "EP":
                        adv = advantages.copy()
                    else: # FP
                        adv = advantages[:, :, agent_id].copy()
                    population_advantages.append(adv)

                # Update actor with population data
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id],  # Pass main buffer as default
                    advantages.copy() if self.state_type == "EP" else advantages[:, :, agent_id].copy(),
                    self.state_type,
                    population_buffers=population_buffers,
                    population_advantages=population_advantages,
                )
            else:
                # =================================================================================
                # Original Non-EA Logic
                # =================================================================================
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )
            
            # Use the main actor for factor calculation, even if EA is used
            actor_to_evaluate = self.actor[agent_id].actor if is_ea_actor else self.actor[agent_id]

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = actor_to_evaluate.evaluate_actions(
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

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = actor_to_evaluate.evaluate_actions(
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

            # update factor for next agent
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

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info