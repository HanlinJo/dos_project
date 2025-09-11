"""HAPPO algorithm with EA guidance."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAPPOEA(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize HAPPO algorithm with EA guidance.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAPPOEA, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        # EA MODIFICATION START: Add hyperparameters for EA guidance
        self.use_ea_guidance = args.get("use_ea_guidance", False)
        self.ea_alpha = args.get("ea_alpha", 0.1)
        # EA MODIFICATION END

    def update(self, sample, elite_agent=None): # EA MODIFICATION: Accept an elite agent
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
            elite_agent: (OnPolicyBase.actor) An elite agent from the EA population for distillation.
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
        action_log_probs, dist_entropy, action_dist = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            return_dist=True # We need the distribution object for KL divergence
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

        # EA MODIFICATION START: Calculate distillation loss
        distillation_loss = torch.tensor(0.0).to(**self.tpdv)
        if self.use_ea_guidance and elite_agent is not None:
            with torch.no_grad():
                # Get action distribution from the elite agent.
                # Note: The elite agent should have a compatible `evaluate_actions` method.
                _, _, elite_action_dist = elite_agent.evaluate_actions(
                    obs_batch,
                    rnn_states_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    active_masks_batch,
                    return_dist=True
                )

            # Calculate KL-divergence: D_KL(elite || current)
            # This pushes the current policy to be more like the elite's policy.
            kl_div = torch.distributions.kl.kl_divergence(elite_action_dist, action_dist)
            
            # Apply active masks to the distillation loss
            if self.use_policy_active_masks:
                distillation_loss = (kl_div * active_masks_batch).sum() / active_masks_batch.sum()
            else:
                distillation_loss = kl_div.mean()
        # EA MODIFICATION END


        self.actor_optimizer.zero_grad()

        # EA MODIFICATION START: Add distillation loss to the total loss
        total_loss = policy_loss - dist_entropy * self.entropy_coef + self.ea_alpha * distillation_loss
        total_loss.backward()
        # EA MODIFICATION END

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    # EA MODIFICATION: The train method must accept the elite agent
    def train(self, actor_buffer, advantages, state_type, elite_agent=None):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
            elite_agent: (OnPolicyBase.actor) An elite agent from the EA population.
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
                # EA MODIFICATION: Pass the elite agent to the update function
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample, elite_agent
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info