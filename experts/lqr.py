'''
Abstract class for LQR and policy format for collecting trajectories.

2025 Matt Taylor
'''

from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
from abc import ABC, abstractmethod
from scipy import linalg
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

class LQR(ABC):
    '''
    Abstract base class for LQR controllers.
    '''
    def __init__(self):
        self.A, self.B = self.define_state_space_matrices()
        self.Q, self.R = self.define_cost_matrices()
        self.K = LQR.lqr(self.A, self.B, self.Q, self.R)

    @abstractmethod
    def define_state_space_matrices(self):
        '''
        Defines state space matrices for the system x_dot = Ax + Bu.
        Returns:
            A, B: State space matrices.
        '''
        pass

    @abstractmethod
    def define_cost_matrices(self):
        '''
        Defines cost matrices for the LQR cost objective.
        Returns:
            Q, R: Cost matrices for state and input.
        '''
        pass

    def get_system(self):
        '''
        Getter function for all important LQR variables.
        '''
        return self.A, self.B, self.Q, self.R, self.K

    @staticmethod
    def get_are(A, B, Q, R):
        '''
        Get solution for continuous algebraic Ricatti equation.
        '''
        P = linalg.solve_continuous_are(A, B, Q, R)
        return P

    @staticmethod
    def lqr(A, B, Q, R):
        '''
        Solve for optimal LQR feedback gain.
        '''
        P = LQR.get_are(A, B, Q, R)
        K = linalg.inv(R) @ B.T @ P
        return K

    @staticmethod
    def get_input(K, x):
        '''
        Get control input from state.
        '''
        u = -np.dot(K, x)
        return u
    
    @abstractmethod
    def policy(self, observation):
        '''
        LQR Policy. First convert the observation to the state. Then apply `get_input`.
        '''
        pass

    @staticmethod
    def check_stability(A, B, K):
        '''
        Prove closed loop stability
        '''
        s = A - B@K
        eigenvalues, eigenvectors = np.linalg.eig(s)
        assert(np.all(eigenvalues.real < 0))

    @staticmethod
    def checkControllable(A, B):
        # function to check if matrix W_cont has full rank
        n = A.shape[0]
        W = B
        for i in range(1, n):
            col = np.linalg.matrix_power(A, i) @ B
            W = np.block([W, col])
        rank = np.linalg.matrix_rank(W)
        if rank == n:
            cond = np.linalg.cond(W)
            print('Controllable with condition number:', cond)
        else:
            raise ValueError('System is not controllable.')
            
class LQRPolicy(LQR, ActorCriticPolicy):
    '''
    Wrapper for LQR expert. Puts policy in a format that the imitation library expects for collecting trajectories.
    This only occurs if env is not None.
    '''
    def __init__(self, env=None):
        LQR.__init__(self)
        if env is not None:
            self.observation_space = env.observation_space
            self.action_space = env.action_space
            lr_schedule = lambda _: 0.
            ActorCriticPolicy.__init__(self, self.observation_space, self.action_space, lr_schedule=lr_schedule)

    def predict(self,
                observation: Union[np.ndarray, Dict[str, np.ndarray]],
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = True,
               ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        
        action = self.policy(observation)
        return action, None