import torch
import numpy as np
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.acquisition import ExpectedImprovement


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
        

        #NOISE_SE = 0.2
        #self.yvar = torch.tensor(NOISE_SE**2)

        self.bound = self.up_bound = 0
        self.alpha = alpha
        self.size = num_points
        self.num_workers = num_workers


    def fit(self):
        '''
        Обучение Гауссовского процесса для набранных точек
        '''
        # копируем из буфера набранные точки 
        self.points = self.points_buffer[:self.up_bound, :self.up_bound].double()
        self.ys = self.ys_buffer[:self.up_bound]
        # фитимся на них
        self.model = SingleTaskGP(train_X=self.points, train_Y=self.ys)
        #self.model = FixedNoiseGP(train_X=self.points, train_Y=self.ys, train_Yvar=self.yvar.expand_as(self.ys))
        

    
    def calc_int_r_and_add_points(self,
                                  z: object,
                                  value_ext: object,
                                  applied: bool):
        '''
        Добавление точек в буффер и подсчет внешней награды для агента
        '''
        
        
        if applied:
            ei = ExpectedImprovement(model=self.model, best_f=value_ext.max())
            with torch.no_grad():
                values = [ei(vector.unsqueeze(0)).numpy() for vector in z]
            int_r = np.stack(values)

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
