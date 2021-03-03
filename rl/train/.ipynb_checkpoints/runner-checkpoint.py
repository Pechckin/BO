import json
import torch
import random
import numpy as np
from tqdm import tqdm
from rl.utils.utils import *
from rl.model.ppo_agent import PPOAgent
from rl.model.model import PPOModel
from rl.utils.buffer import Buffer
from rl.train.trainer import Trainer
from rl.utils.logger import Logger
from rl.utils.utils import AttrDict
from rl.envs.multiprocessing_envs import *
from rl.bo.bo import BO


class Runner:
    '''
    Главный ранер всего обучения
    '''

    def __init__(self, config_path: str):
        # достаем конфиг по указанному пути
        with open(f'{config_path}.json') as json_file:
            config = json.load(json_file)

        self.c = AttrDict(config)

        model = PPOModel(action_size=self.c.action_size,
                         lr_optim=self.c.lr_optim,
                         bo_points_dim=self.c.bo_points_dim,
                         device=self.c.device)

        agent = PPOAgent(model)
        buffer = Buffer(self.c.buffer_size, self.c.num_workers, self.c.device)

        self.logger = Logger(self.c.save_dir)
        self.logger.log(self.c, 'params')
        self.envs = [make_env(i, self.c.env) for i in range(self.c.num_workers)]
        self.envs = SubprocVecEnv(self.envs)

        self.trainer = Trainer(agent=agent,
                               buffer=buffer,
                               logger=self.logger,
                               num_batches=self.c.num_batches,
                               batch_size=self.c.batch_size,
                               gamma=self.c.gamma,
                               eps=self.c.eps,
                               clip=self.c.clip,
                               entropy_coef=self.c.entropy_coef,
                               device=self.c.device,
                               int_coef=self.c.int_coef,
                               ext_coef=self.c.ext_coef)

        # уменьшение лернинг рейта и клиппинга для ppo
        self.reduce_lr = lambda f: f * self.c.lr_optim
        self.reduce_eps = lambda f: f * self.c.eps
        self.device = self.c.device
        self.reset()

        self.bo = BO(num_points=self.c.buffer_size * self.c.num_workers * self.c.tay_coef,
                     points_dim=self.c.bo_points_dim,
                     num_workers=self.c.num_workers,
                     alpha=self.c.alpha)

    def reset(self):
        self.test_rewards = []

    # запускаем тест
    def run_episode(self, total_steps, test=False):
        env = [make_env(999, self.c.env)]
        env = SubprocVecEnv(env)
        state = env.reset()

        total_reward = 0
        episode_done, episode_reward = False, 0

        while not episode_done:
            action, log_probs = self.trainer.agent.action(state)
            next_state, reward, done, infos = env.step(action)

            total_reward += reward
            state = next_state

            for info in infos:
                if 'episode' in info.keys():
                    episode_done = True
                    episode_reward = info['episode']['r']
                    break

        self.test_rewards.append(episode_reward)
        env.close()

    def run(self):
        total_steps = 0
        applied = False
        state = self.envs.reset()

        bar = tqdm(total=self.c.train_steps, position=0, leave=False,
                   desc=f'test score {0.0}, steps {0}')

        while total_steps <= self.c.train_steps:
            # уменьшаем лернинг рейт и клиппинг ppo
            frac = 1.0 - ((total_steps + 1) - 1.0) / self.c.train_steps
            reduced_lr = self.reduce_lr(frac)
            eps = self.reduce_eps(frac)

            # совершаем действие по политике
            action, log_probs = self.trainer.agent.action(state)
            next_state, reward, done, infos = self.envs.step(action)

            # считаем value функцию
            with torch.no_grad():
                _, value_int, value_ext = self.trainer.agent.model.NET(state.to(self.device))
                _, next_value_int, next_value_ext = self.trainer.agent.model.NET(next_state.to(self.device))
                value_int, value_ext = value_int.cpu(), value_ext.cpu()
                next_value_int, next_value_ext = next_value_int.cpu(), next_value_ext.cpu()

            # получаем латентное представление состояния
            z = self.trainer.agent.model.AU.get_repr(next_state.to(self.device))
            z = z.cpu()
            # считаем внешнюю награду
            ei = self.bo.calc_int_r_and_add_points(z, next_value_ext, applied)

            curiosity_reward = torch.FloatTensor(ei).cpu()
            reward = torch.FloatTensor(reward).unsqueeze(1)
            mask = torch.FloatTensor(1 - done).unsqueeze(1)
            # отправляем транзицию в буффер
            self.trainer.buffer.push(state, value_int, value_ext, action,
                                     next_value_int, next_value_ext, reward, mask,
                                     log_probs, curiosity_reward)
            # обновляем байесовскую оптимизацию
            if self.trainer.buffer.full() and total_steps > self.c.bo_warm_up:
                self.bo.fit()
                applied = True
            # обновляем все остальное
            self.trainer.train(eps, total_steps)
            state = next_state
            total_steps += self.c.num_workers
            # тестирование
            if not total_steps % self.c.test_every:
                for j in range(self.c.test_episodes):
                    self.run_episode(total_steps, test=True)

                self.logger.log({'mean_test_reward': np.mean(self.test_rewards),
                                 'std_test_reward': np.std(self.test_rewards),
                                 'total_steps_test': total_steps}, 'Test')

                bar.set_description(
                    f'test score {np.mean(self.test_rewards)}, steps {total_steps}')
                bar.update(self.c.test_every)
                self.test_rewards = []
                self.logger.save()

            self.trainer.agent.model.optimizer.param_groups[0]['lr'] = reduced_lr
        self.logger.save()
        self.envs.close()
