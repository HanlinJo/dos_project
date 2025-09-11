import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.models.base.mlp import MLPBase
from harl.models.base.rnn import RNNLayer
# from harl.utils.models_tools import get_activate_func, get_init_method
from harl.utils.models_tools import get_active_func, get_init_method
class EA_HA_Agent_SR(nn.Module):
    """
    EA 异构智能体的共享特征提取网络 (Shared Representation)。
    对于每个智能体类型，都会有一个独立的SR网络实例。
    这部分网络负责从高维观测中提取有用的特征。
    """
    def __init__(self, args, obs_dim, act_dim, device):
        super(EA_HA_Agent_SR, self).__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        #
        #
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_size = args.hidden_size
        self.recurrent_n = args.recurrent_n
        self.use_feature_normalization = args.use_feature_normalization
        self.use_recurrent_policy = args.use_recurrent_policy

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(self.obs_dim)

        self.base = MLPBase(args, self.obs_dim)
        
        if self.use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self.recurrent_n, args.use_orthogonal)

    def forward(self, obs, rnn_states):
        if self.use_feature_normalization:
            obs = self.feature_norm(obs)
        
        actor_features = self.base(obs)

        if self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states)

        return actor_features, rnn_states

class EA_HA_Agent_W(nn.Module):
    """
    EA 异构智能体的独立决策网络 (Weights)。
    每个种群个体都有自己独立的W网络，EA的进化过程主要作用于这部分网络。
    """
    def __init__(self, args, act_dim, device):
        super(EA_HA_Agent_W, self).__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.act_dim = act_dim
        self.hidden_size = args["hidden_size"]

        self.act = nn.Linear(self.hidden_size, self.act_dim)

    def forward(self, actor_features):
        action_logits = self.act(actor_features)
        return action_logits