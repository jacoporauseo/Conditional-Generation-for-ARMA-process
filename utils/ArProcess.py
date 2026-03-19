import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.stats import t as student_t
from typing import Tuple, List

class AR1(ABC):   
    @abstractmethod
    def generate_trajectory(self, T: int):
        "Create a trajectory from the AR process"
        pass
    def conditional_pdf(self, x: np.ndarray, c:float):
        """Conditional pdf given x_{t-1}"""
        pass
    def sample_prev(self,T:int): 
        """Return an array of shape [T,2] with [x_t,x_{t-1}]"""
        pass 


class AR_normal(AR1):
    """Class for AR1 process using Normal innovations"""
    def __init__(self, phi=0.8, sigma=1, seed = 123):
        self.phi = phi
        self.sigma = sigma  # standard deviation

    def generate_trajectory(self, T:int=100000):
        x = np.zeros(T)
        x[0] = np.random.normal(0, self.sigma)
        for t in range(1, T):
            x[t] = self.phi * x[t-1] + np.random.normal(0, self.sigma)
        return x.reshape(-1, 1)

    def conditional_pdf(self, x: np.ndarray, c: float):
        mu = self.phi * c                          
        x_grid = np.linspace(-4, 4, 100)
        p = (1 / (np.sqrt(2 * np.pi) * self.sigma) * np.exp(-(x_grid - mu)**2 / (2 * self.sigma**2)))  
        return x_grid, p

    def sample_prev(self, T):
        x = self.generate_trajectory(T = T + 1)
        x = np.concatenate([x[1:], x[:-1]], axis=1)
        return x 

class AR_studentt(AR1):
    """Class for AR1 process using student t innovations"""
    def __init__(self, phi=0.8, scale=1, df=5, seed=123):
        self.phi = phi
        self.scale = scale
        self.df = df
        self.seed = seed

    def generate_trajectory(self, T: int = 100000):
        np.random.seed(self.seed)
        x = np.zeros(T)
        x[0] = np.random.standard_t(self.df) * self.scale
        for t in range(1, T):
            x[t] = self.phi * x[t-1] + np.random.standard_t(self.df) * self.scale
        return x.reshape(-1, 1)

    def conditional_pdf(self, x: np.ndarray, c: float):
        mu = self.phi * c
        x_grid = np.linspace(-4, 4, 100)
        p = student_t.pdf(x_grid, df=self.df, loc=mu, scale=self.scale)
        return x_grid, p

    def sample_prev(self, T):
        x = self.generate_trajectory(T=T + 1)
        x = np.concatenate([x[1:], x[:-1]], axis=1)
        return x

