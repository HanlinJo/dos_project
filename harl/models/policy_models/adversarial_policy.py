# harl/models/policy_models/adversarial_policy.py

import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act import ACTLayer
from harl.utils.envs_tools import get_shape_from_obs_space


class AdversarialPolicy(nn.Module):
    """Adversarial policy model. Outputs attack actions given victim's observations."""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize AdversarialPolicy model."""
        super(AdversarialPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        # victim_obs_shape = get_shape_from_obs_space(victim_obs_space)
        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.to(device)

    def forward(
        self, victim_obs, rnn_states, masks, available_actions=None, deterministic=False
    ):
        """Compute actions from the given inputs."""
        victim_obs = check(victim_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(victim_obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )

        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self, victim_obs, rnn_states, action, masks, available_actions=None, active_masks=None
    ):
        """Compute action log probability and action distribution entropy."""
        victim_obs = check(victim_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(victim_obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
        )

        return action_log_probs, dist_entropy, action_distribution