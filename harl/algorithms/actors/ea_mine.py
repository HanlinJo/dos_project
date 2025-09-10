import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def fenchel_dual_loss(l, m, measure='JSD'):
    """Computes the f-divergence distance between positive and negative joint distributions."""
    N, units = l.size()
    u = torch.mm(m, l.t())
    mask = torch.eye(N, device=l.device)
    n_mask = 1 - mask
    
    log_2 = math.log(2.)
    if measure == 'JSD':
        E_pos = log_2 - F.softplus(-u)
        E_neg = F.softplus(-u) + u - log_2
    else:  # Fallback to GAN
        E_pos = -F.softplus(-u)
        E_neg = F.softplus(-u) + u

    # A more stable implementation of the loss
    pos_term = -E_pos.diag().mean()
    neg_term = (E_neg * n_mask).sum() / (N * (N - 1))
    loss = pos_term + neg_term
    return loss

class MINE(nn.Module):
    """
    Mutual Information Neural Estimator (MINE).
    """
    def __init__(self, x_dim, y_dim, hidden_size=128, measure="JSD"):
        super(MINE, self).__init__()
        self.measure = measure
        self.l1 = nn.Linear(x_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(y_dim, hidden_size)

    def forward(self, x, y):
        # Detach y to ensure gradients only flow through x's encoder
        y_ = y.detach()
        em_1 = F.leaky_relu(self.l1(x))
        em_1 = F.leaky_relu(self.l2(em_1))
        em_2 = F.leaky_relu(self.l3(y_))
        loss = fenchel_dual_loss(em_1, em_2, measure=self.measure)
        return loss