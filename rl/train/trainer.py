import torch.nn.functional as F
from rl.model.model import PPOModel
from rl.model.ppo_agent import PPOAgent
from rl.utils.buffer import Buffer
from rl.utils.logger import Logger
import torch
from torch import nn as nn
import numpy as np
from torch.distributions import Categorical


class Trainer:
    '''
    Обучение модели
    '''

    def __init__(
            self,
            agent: PPOAgent,
            buffer: Buffer,
            logger: object,
            num_batches: int,
            batch_size: int,
            gamma: float,
            eps: float,
            clip: float,
            entropy_coef: float,
            device: str,
            int_coef: float,
            ext_coef: float,
    ):

        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.buffer = buffer
        self.device = device

        self.clip = clip
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.eps = eps

        self.int_coef = int_coef
        self.ext_coef = ext_coef

        self.forward_mse = nn.MSELoss()

    def train(self, eps, total_steps):
        if not self.buffer.full():
            return
        # семплируем несколько траекторий
        losses = {'loss': [],
                  'actor_loss': [],
                  'critic_loss': [],
                  'entropy_loss': [],
                  'forward_loss': []}
        self.eps = eps
        for batch in self.buffer.sample(self.num_batches, self.batch_size):
            # тренеровка автоэнкодера
            s1, s2 = self.agent.model.AU(batch.s)
            forward_loss = self.forward_mse(s1, s2)

            # ниже классический ppo
            actor_s, critic_si, critic_se = self.agent.model.NET(batch.s)

            A = (batch.ri - batch.vi) * self.int_coef + (batch.re - batch.ve) * self.ext_coef
            A = (A - A.mean()) / (A.std() + 1e-8)

            probs = F.softmax(actor_s, dim=1)
            log_probs = F.log_softmax(actor_s, dim=1)
            dist_entropy = -(log_probs * probs).sum(-1).mean()

            actions_long = batch.a.long()

            log_probs = log_probs.gather(1, actions_long)
            old_log_probs = batch.lp.squeeze().gather(1, actions_long)

            ratio = (log_probs - old_log_probs).exp()
            L1 = ratio * A
            L2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * A
            actor_loss = -torch.min(L1, L2).mean()

            critic_loss_i = self.agent.model.loss_f(batch.ri, critic_si)
            critic_loss_e = self.agent.model.loss_f(batch.re, critic_se)

            critic_loss = (critic_loss_i + critic_loss_e) * 0.5
            # суммарный лосс
            loss = critic_loss + actor_loss - dist_entropy * self.entropy_coef + forward_loss

            self.agent.model.optimizer.zero_grad()

            losses['loss'].append(loss.cpu().item())
            losses['actor_loss'].append(actor_loss.cpu().item())
            losses['critic_loss'].append(critic_loss.cpu().item())
            losses['entropy_loss'].append(dist_entropy.cpu().item())
            losses['forward_loss'].append(forward_loss.cpu().item())

            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(self.agent.model.NET.parameters()) +
                                           list(self.agent.model.AU.parameters()), self.clip)

            self.agent.model.optimizer.step()

        # store
        for key in losses:
            losses[key] = np.mean(losses[key])
        self.logger.log({'loss': losses['loss'],
                         'actor_loss': losses['actor_loss'],
                         'critic_loss': losses['critic_loss'],
                         'entropy_loss': losses['entropy_loss'],
                         'forward_loss': losses['forward_loss']}, 'Train')
