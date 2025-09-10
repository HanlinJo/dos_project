# """Runner for on-policy HARL algorithms."""
# import numpy as np
# import torch
# from harl.utils.trans_tools import _t2n
# from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


# class OnPolicyHARunner(OnPolicyBaseRunner):
#     """Runner for on-policy HA algorithms."""

#     def train(self):
#         """Train the model."""
#         actor_train_infos = []

#         # factor is used for considering updates made by previous agents
#         factor = np.ones(
#             (
#                 self.algo_args["train"]["episode_length"],
#                 self.algo_args["train"]["n_rollout_threads"],
#                 1,
#             ),
#             dtype=np.float32,
#         )

#         # compute advantages
#         if self.value_normalizer is not None:
#             advantages = self.critic_buffer.returns[
#                 :-1
#             ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
#         else:
#             advantages = (
#                 self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
#             )

#         # normalize advantages for FP
#         if self.state_type == "FP":
#             active_masks_collector = [
#                 self.actor_buffer[i].active_masks for i in range(self.num_agents)
#             ]
#             active_masks_array = np.stack(active_masks_collector, axis=2)
#             advantages_copy = advantages.copy()
#             advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
#             mean_advantages = np.nanmean(advantages_copy)
#             std_advantages = np.nanstd(advantages_copy)
#             advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

#         if self.fixed_order:
#             agent_order = list(range(self.num_agents))
#         else:
#             agent_order = list(torch.randperm(self.num_agents).numpy())
#         for agent_id in agent_order:
#             self.actor_buffer[agent_id].update_factor(
#                 factor
#             )  # current actor save factor

#             # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
#             available_actions = (
#                 None
#                 if self.actor_buffer[agent_id].available_actions is None
#                 else self.actor_buffer[agent_id]
#                 .available_actions[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
#             )

#             # compute action log probs for the actor before update.
#             old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
#                 self.actor_buffer[agent_id]
#                 .obs[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
#                 self.actor_buffer[agent_id]
#                 .rnn_states[0:1]
#                 .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
#                 self.actor_buffer[agent_id].actions.reshape(
#                     -1, *self.actor_buffer[agent_id].actions.shape[2:]
#                 ),
#                 self.actor_buffer[agent_id]
#                 .masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
#                 available_actions,
#                 self.actor_buffer[agent_id]
#                 .active_masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
#             )

#             # update actor
#             if self.state_type == "EP":
#                 actor_train_info = self.actor[agent_id].train(
#                     self.actor_buffer[agent_id], advantages.copy(), "EP"
#                 )
#             elif self.state_type == "FP":
#                 actor_train_info = self.actor[agent_id].train(
#                     self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
#                 )

#             # compute action log probs for updated agent
#             new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
#                 self.actor_buffer[agent_id]
#                 .obs[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
#                 self.actor_buffer[agent_id]
#                 .rnn_states[0:1]
#                 .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
#                 self.actor_buffer[agent_id].actions.reshape(
#                     -1, *self.actor_buffer[agent_id].actions.shape[2:]
#                 ),
#                 self.actor_buffer[agent_id]
#                 .masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
#                 available_actions,
#                 self.actor_buffer[agent_id]
#                 .active_masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
#             )

#             # update factor for next agent
#             factor = factor * _t2n(
#                 getattr(torch, self.action_aggregation)(
#                     torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
#                 ).reshape(
#                     self.algo_args["train"]["episode_length"],
#                     self.algo_args["train"]["n_rollout_threads"],
#                     1,
#                 )
#             )
#             actor_train_infos.append(actor_train_info)

#         # update critic
#         critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

#         return actor_train_infos, critic_train_info

# """Runner for on-policy HARL algorithms."""
# import numpy as np
# import torch
# import copy
# from harl.utils.trans_tools import _t2n
# from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
# # EA MODIFICATION: Import EA_HAPPO
# from harl.algorithms.actors.happo import HAPPO
# from harl.algorithms.critics.v_critic import VCritic
# from harl.algorithms.actors.ea_happo import EA_HAPPO

# class OnPolicyHARunner(OnPolicyBaseRunner):
#     """Runner for on-policy HA algorithms."""

#     # FIX: Correct the constructor to accept three arguments as called by train.py
#     def __init__(self, args, algo_args, env_args):
#         """
#         Initialize the OnPolicyHARunner.
#         Args:
#             args: command-line arguments.
#             algo_args: arguments for the algorithm.
#             env_args: arguments for the environment.
#         """
#         # Combine the dictionaries into a single config dictionary
#         # config = {}
#         # config.update(args)
#         # config.update(algo_args)
#         # config.update(env_args)
#         # print(algo_args, env_args)
#         # Call the parent constructor with the combined config
#         super(OnPolicyHARunner, self).__init__(args, algo_args, env_args)

#         # EA MODIFICATION: Conditionally initialize EA_HAPPO or the original HAPPO+VCritic
#         if self.all_args['use_ea']:
#             print("--- Using Evolutionary Algorithm (EA) assisted HAPPO ---")
#             # When using EA, the EA_HAPPO class encapsulates both actor and critic logic
#             self.algo = EA_HAPPO(self.all_args, self.envs.observation_space, self.envs.action_space, device=self.device)
#             # For compatibility with existing methods, we can still assign actor and critic references
#             self.actor = self.algo
#             self.critic = self.algo.critic # Point to the critic inside EA_HAPPO
#         else:
#             # Original initialization for standard HAPPO
#             self.actor = HAPPO(self.all_args, self.envs.observation_space, self.envs.action_space, device=self.device)
#             self.critic = VCritic(self.all_args, self.envs.share_observation_space, self.device)
#             self.algo = (self.actor, self.critic) # Keep the tuple structure for the original case

#     def train(self):
#         """Train the model."""
#         # EA MODIFICATION: If using EA, call its unified train method. Otherwise, use the original logic.
#         if self.all_args.use_ea:
#             # The EA_HAPPO class handles the entire training process for all agents and critics
#             train_info = self.algo.train(self.actor_buffer, self.critic_buffer)
#             # The train_info dictionary is returned for logging
#             return train_info

#         # --- ORIGINAL HAPPO TRAINING LOGIC ---
#         actor_train_infos = []

#         # factor is used for considering updates made by previous agents
#         factor = np.ones(
#             (
#                 self.algo_args["train"]["episode_length"],
#                 self.algo_args["train"]["n_rollout_threads"],
#                 1,
#             ),
#             dtype=np.float32,
#         )

#         # compute advantages
#         if self.value_normalizer is not None:
#             advantages = self.critic_buffer.returns[
#                 :-1
#             ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
#         else:
#             advantages = (
#                 self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
#             )

#         # normalize advantages for FP
#         if self.state_type == "FP":
#             active_masks_collector = [
#                 self.actor_buffer[i].active_masks for i in range(self.num_agents)
#             ]
#             active_masks_array = np.stack(active_masks_collector, axis=2)
#             advantages_copy = advantages.copy()
#             advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
#             mean_advantages = np.nanmean(advantages_copy)
#             std_advantages = np.nanstd(advantages_copy)
#             advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

#         for agent_id in range(self.num_agents):
#             if self.state_type == "EP":
#                 self.actor_buffer[agent_id].compute_advantages(
#                     self.critic_buffer.returns[:-1], self.value_normalizer
#                 )
#                 adv = self.actor_buffer[agent_id].advantages
#             else:
#                 adv = advantages

#             # update actor
#             actor_train_info = self.actor[agent_id].train(
#                 self.actor_buffer[agent_id], adv, factor
#             )
#             old_actions_logprob = self.actor_buffer[agent_id].actions_logprob
#             # new_actions_logprob is the log probability of the actions taken by the agent,
#             # but evaluated by the new policy
#             (
#                 _,
#                 new_actions_logprob,
#                 _,
#                 _,
#             ) = self.actor[agent_id].evaluate_actions(
#                 self.actor_buffer[agent_id]
#                 .obs[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
#                 self.actor_buffer[agent_id]
#                 .rnn_states[0:1]
#                 .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
#                 self.actor_buffer[agent_id].actions.reshape(
#                     -1, *self.actor_buffer[agent_id].actions.shape[2:]
#                 ),
#                 self.actor_buffer[agent_id]
#                 .masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
#                 None,
#                 self.actor_buffer[agent_id]
#                 .active_masks[:-1]
#                 .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
#             )

#             # update factor for next agent
#             factor = factor * _t2n(
#                 getattr(torch, self.action_aggregation)(
#                     torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
#                 ).reshape(
#                     self.algo_args["train"]["episode_length"],
#                     self.algo_args["train"]["n_rollout_threads"],
#                     1,
#                 )
#             )
#             actor_train_infos.append(actor_train_info)

#         # update critic
#         critic_train_info = self.critic.train(self.critic_buffer, self.actor_buffer)

#         return actor_train_infos, critic_train_info
    
#     # EA MODIFICATION: Override the run method to add population evaluation and evolution
#     def run(self):
#         """Run the training and evaluation loop."""
#         self.warmup()

#         for episode in range(self.episodes):
#             if self.algo_args["train"]["use_linear_lr_decay"]:
#                 if self.all_args.use_ea: # Decay LR for EA_HAPPO
#                     self.algo.lr_decay(self.total_num_steps, self.num_env_steps)
#                 else: # Original decay logic
#                     self.actor.lr_decay(self.total_num_steps, self.num_env_steps)
#                     self.critic.lr_decay(self.total_num_steps, self.num_env_steps)

#             # collection phase
#             obs, rnn_states_actor, rnn_states_critic, actions, actions_logprob, \
#             values, rewards, masks, active_masks, state = self.collect(episode)

#             # store phase
#             self.store(obs, state, rnn_states_actor, rnn_states_critic, actions, actions_logprob, values, rewards, masks)

#             # train phase
#             train_infos = self.train()
            
#             # log phase
#             self.total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
#             if episode % self.log_interval == 0:
#                 self.log_train(train_infos, self.total_num_steps)

#             # eval phase
#             if episode % self.eval_interval == 0 and self.use_eval:
#                 self.eval(self.total_num_steps)
                
#             # EA MODIFICATION: Add population evaluation and evolution step
#             if self.all_args.use_ea and (episode % self.all_args.ea_evolution_interval == 0):
#                 print(f"\n--- Episode {episode}: Starting EA Population Evaluation and Evolution ---")
#                 fitness_scores = self.evaluate_population()
#                 self.algo.evolve_population(fitness_scores)
#                 # Log fitness scores
#                 for agent_id in range(self.num_agents):
#                     agent_fitness = [score[agent_id] for score in fitness_scores]
#                     print(f"Agent Type {agent_id} Fitness Scores: {agent_fitness}")
#                     if self.writer:
#                         self.writer.add_scalar(f"eval_fitness/agent{agent_id}_mean_fitness", np.mean(agent_fitness), self.total_num_steps)

#     # EA MODIFICATION: Add a method to evaluate the population fitness
#     @torch.no_grad()
#     def evaluate_population(self):
#         """
#         Evaluates the fitness (average return) of each member in the population.
#         """
#         fitness_scores = []  # List to store fitness scores for each population member
#         num_eval_episodes = 3  # Number of episodes to average over for each member

#         # Iterate over each population member (pop_id from 1 to pop_size)
#         for pop_id in range(1, self.all_args.pop_size + 1):
#             pop_member_rewards = [0.0 for _ in range(self.num_agents)]
#             for _ in range(num_eval_episodes):
#                 # We can reuse the envs from the runner for evaluation
#                 obs, _, _ = self.envs.reset()
                
#                 rnn_states_actor = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_n, self.hidden_size), dtype=np.float32)
#                 masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

#                 episode_rewards = np.zeros(self.num_agents)
                
#                 for step in range(self.episode_length):
#                     actions, rnn_states_actor = self.collect_eval_actions(obs, rnn_states_actor, masks, pop_id)
                    
#                     obs, rewards, dones, infos = self.envs.step(actions)
#                     # Sum rewards for each agent across all parallel environments
#                     episode_rewards += np.mean(rewards, axis=0).flatten()[:self.num_agents]
                    
#                     # Handle dones for recurrent states and masks
#                     dones_env = np.all(dones, axis=1)
#                     rnn_states_actor[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_n, self.hidden_size), dtype=np.float32)
#                     masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
#                     masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

#                 for i in range(self.num_agents):
#                     pop_member_rewards[i] += episode_rewards[i]
            
#             # Average rewards over evaluation episodes
#             avg_rewards = [r / num_eval_episodes for r in pop_member_rewards]
#             fitness_scores.append(avg_rewards) # Store the average reward as fitness
            
#         return fitness_scores

#     # EA MODIFICATION: Add helper to get actions for a specific population member
#     def collect_eval_actions(self, obs, rnn_states_actor, masks, pop_id):
#         """Helper function to get actions from a specific population member during evaluation."""
#         actions_collector = []
#         rnn_states_actor_collector = []

#         for agent_id in range(self.num_agents):
#             # Use the algo.get_actions method with the specified pop_id
#             action, rnn_state = self.algo.get_actions_for_eval(
#                 obs[:, agent_id],
#                 rnn_states_actor[:, agent_id],
#                 masks[:, agent_id],
#                 agent_id=agent_id,
#                 pop_id=pop_id,
#                 deterministic=True
#             )
#             actions_collector.append(_t2n(action))
#             rnn_states_actor_collector.append(_t2n(rnn_state))

#         actions = np.concatenate(actions_collector, axis=-1)
#         rnn_states_actor_out = np.array(rnn_states_actor_collector).transpose(1, 0, 2, 3)

#         return actions, rnn_states_actor_out

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

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                .available_actions[:-1]
                .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
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

            # update actor
            if self.state_type == "EP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages.copy(), "EP"
                )
            elif self.state_type == "FP":
                actor_train_info = self.actor[agent_id].train(
                    self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                )

            # compute action log probs for updated agent
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