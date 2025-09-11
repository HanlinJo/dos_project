"""HAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase
import random
import fastrand
import math

class EAHAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(EAHAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        # EA parameters
        self.use_ea = args.get("use_ea", False)
        if self.use_ea:
            self.pop_size = args["pop_size"]
            self.elite_fraction = args["elite_fraction"]
            self.num_elitists = int(self.elite_fraction * self.pop_size)
            if self.num_elitists < 1: self.num_elitists = 1
            self.crossover_prob = args["crossover_prob"]
            self.mutation_prob = args["mutation_prob"]
            self.prob_reset_and_sup = args["prob_reset_and_sup"]
            self.frac = args["frac"]
            self.EA_alpha = args["EA_alpha"]
            self.Org_alpha = args["Org_alpha"]

            # Create a population of actors
            self.population = [self.actor] + [
                type(self.actor)(self.args, self.obs_space, self.act_space, self.device)
                for _ in range(self.pop_size - 1)
            ]
            self.optimizers = [
                torch.optim.Adam(
                    actor.parameters(), lr=self.lr, eps=self.opti_eps
                )
                for actor in self.population
            ]
    def get_fitness(self, advantages):
        """Calculate fitness for each individual in the population."""
        return advantages.mean().item()

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        """Select offspring using tournament selection."""
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))
        if len(offsprings) % 2 != 0:
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def crossover_inplace(self, gene1, gene2):
        """Perform crossover between two parent genes."""
        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):
            W1 = param1.data
            W2 = param2.data
            if len(W1.shape) == 2:
                num_variables = W1.shape[0]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)
                for i in range(num_cross_overs):
                    receiver_choice = random.random()
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])
                        W2[ind_cr, :] = W1[ind_cr, :]

    def mutate_inplace(self, gene):
        """Perform mutation on a gene."""
        mut_strength = 0.1
        super_mut_strength = 10
        super_mut_prob = self.prob_reset_and_sup
        reset_prob = super_mut_prob + self.prob_reset_and_sup

        for param in gene.parameters():
            W = param.data
            if len(W.shape) == 2:
                num_variables = W.shape[0]
                for i in range(num_variables):
                    if random.random() < self.mutation_prob:
                        index_list = random.sample(
                            range(W.shape[1]), int(W.shape[1] * self.frac)
                        )
                        random_num = random.random()
                        if random_num < super_mut_prob:
                            for ind in index_list:
                                W[i, ind] += random.gauss(
                                    0, super_mut_strength * abs(W[i, ind].item())
                                )
                        elif random_num < reset_prob:
                            for ind in index_list:
                                W[i, ind] = random.gauss(0, 1)
                        else:
                            for ind in index_list:
                                W[i, ind] += random.gauss(
                                    0, mut_strength * abs(W[i, ind].item())
                                )
            W = torch.clamp(W, -1000000, 1000000)

    def clone(self, master, replacee):
        """Clone master's weights to replacee."""
        for target_param, source_param in zip(
            replacee.parameters(), master.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def evolve(self, fitness_evals):
        """Evolve the population of actors."""
        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[: self.num_elitists]

        offsprings = self.selection_tournament(
            index_rank,
            num_offsprings=len(index_rank) - self.num_elitists,
            tournament_size=3,
        )

        unselects = []
        for i in range(self.pop_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism
        for i in elitist_index:
            if unselects:
                replacee = unselects.pop(0)
                self.clone(self.population[i], self.population[replacee])

        # Crossover
        if len(unselects) % 2 != 0:
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            if random.random() < self.crossover_prob:
                self.crossover_inplace(self.population[i], self.population[j])

        # Mutation
        for i in range(self.pop_size):
            if i not in elitist_index:
                self.mutate_inplace(self.population[i])
    def update(self, sample, actor, optimizer):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
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

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, _ = actor.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
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

        optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(actor.parameters())

        optimizer.step()

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

        if self.use_ea:
            # Evaluate fitness of each actor in the population
            fitness_evals = []
            for actor in self.population:
                # Note: This is a simplified fitness evaluation. A more robust
                # implementation would involve running each actor in the environment
                # for a number of episodes and averaging the returns.
                # For now, we use the advantage from the main actor's buffer as a proxy.
                fitness_evals.append(self.get_fitness(advantages))
            
            # Evolve the population
            self.evolve(fitness_evals)

        # Train each actor in the population
        num_actors_to_train = self.pop_size if self.use_ea else 1
        actors_to_train = self.population if self.use_ea else [self.actor]
        optimizers_to_use = self.optimizers if self.use_ea else [self.actor_optimizer]

        for actor, optimizer in zip(actors_to_train, optimizers_to_use):
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
                        sample, actor, optimizer
                    )

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch * num_actors_to_train

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info