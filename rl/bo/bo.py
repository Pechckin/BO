import torch
import numpy as np


class BO:
    '''
    Байесовская оптимизация
    '''

    def __init__(self,
                 num_points: int,
                 points_dim: int,
                 num_workers: int,
                 alpha: float):
        # выделяем память под все точки и под накопительный буффер
        self.points = torch.zeros(num_points, points_dim, dtype=torch.float64)
        self.points_buffer = torch.zeros(num_points, points_dim, dtype=torch.float64)
        # тоно также для значений функции
        self.ys = torch.zeros(num_points, 1, dtype=torch.float64)
        self.ys_buffer = torch.zeros(num_points, 1, dtype=torch.float64)
        # память под ковариационную матрицу
        self.inv_K = torch.zeros(num_points, num_points, dtype=torch.float64)
        self.K = torch.zeros(num_points, num_points, dtype=torch.float64)

        self.bound = self.up_bound = 0
        self.alpha = alpha
        self.size = num_points
        self.num_workers = num_workers

    @staticmethod
    def l2_distance(v, u, scaling=1.0):
        shape = (len(v), len(u), -1)
        u = u.unsqueeze(0).expand(*shape)
        v = v.unsqueeze(1).expand(*shape)
        return ((scaling * (v - u)) ** 2).sum(dim=-1)

    @staticmethod
    def l1_distance(v, u, scaling=1.0):
        shape = (len(v), len(u), -1)
        u = u.unsqueeze(0).expand(*shape)
        v = v.unsqueeze(1).expand(*shape)
        return scaling * (v - u).abs().sum(dim=-1)

    @staticmethod
    def exponential_kernel(v, u, l_scaling=1.0, distance_l=2):
        if distance_l == 2:
            d = self.l2_distance(v, u, l_scaling)
        else:
            d = self.l1_distance(v, u, l_scaling)
        return torch.exp(-d)

    def fit(self):
        '''
        Обучение Гауссовского процесса для набранных точек
        '''
        # копируем из буфера набранные точки 
        self.points = self.points_buffer[:self.up_bound, :self.up_bound].double()
        self.ys = self.ys_buffer[:self.up_bound]
        # фитимся на них
        self.K = exponential_kernel(self.points, self.points)
        self.K[np.diag_indices(self.K.shape[0])] += self.alpha
        self.inv_K = self.K.inverse()

    def calc_int_r_and_add_points(self,
                                  z: object,
                                  value_ext: object,
                                  applied: bool):
        '''
        Добавление точек в буффер и подсчет внешней награды для агента
        '''
        if applied:
            K_star = exponential_kernel(z, self.points)
            mlt_inv = torch.matmul(self.inv_K, self.ys)
            mlt_star = torch.matmul(K_star, self.inv_K)

            mu = torch.matmul(K_star, mlt_inv).squeeze()
            sigma = 1.0 - (K_star * mlt_star).sum(-1)

            y_target = torch.max(self.ys)  # + 0.001
            mu_target = mu - y_target
            proba = 0.5 * (1 + torch.erf(mu_target / sigma / 2 ** 0.5))
            int_r = proba.float().view(-1, 1)
        else:
            int_r = np.zeros((self.num_workers, 1))

        self.add_points(z, value_ext)
        return int_r

    def add_points(self, z, value_ext):
        '''
        Непосредственное добавление точек в буффер
        '''
        lb, rb = self.bound, self.bound + self.num_workers
        self.points_buffer[lb: rb] = z
        self.ys_buffer[lb: rb] = value_ext
        self.bound = rb % self.size
        self.up_bound = max(self.up_bound, rb)
