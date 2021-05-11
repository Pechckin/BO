import torch
from torch import nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
from rl.utils.nets import ActorCritic, AutoEncoder


class PPOModel(nn.Module):

    def __init__(self, action_size: int,
                 lr_optim: float,
                 bo_points_dim: int,
                 device: str):
        super(PPOModel, self).__init__()
        self.device = device
        # заводим актор-критик сеть

        self.NET = ActorCritic(action_size).apply(self.weights).to(device)
        # автонкодер
        self.AU = AutoEncoder(bo_points_dim).to(device)

        self.optimizer = optim.Adam(list(self.NET.parameters()) + list(self.AU.parameters()), lr=lr_optim, eps=1e-5)
        self.loss_f = nn.MSELoss()

    @staticmethod
    def weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)

    def act(self, state):
        with torch.no_grad():
            actor_out, _, _ = self.NET(state.to(self.device))
        actor_out = actor_out.cpu()
        probs = F.softmax(actor_out, dim=1)
        distribution = Categorical(probs)
        action = distribution.sample()
        return action.numpy(), torch.log(probs)
