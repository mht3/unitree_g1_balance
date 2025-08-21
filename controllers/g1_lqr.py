import numpy as np
from .lqr import LQRPolicy

class G1LQR(LQRPolicy):
    '''
    LQR For Unitree G1 2 Leg Balance Environment.
    '''
    def __init__(self, env=None):
        super().__init__(env)

    def define_state_space_matrices(self):
        '''
        Defines state space matrices for the system x_dot = Ax + Bu.
        Returns:
            A, B: State space matrices.
        '''
        if self.mass_type == 'point':
            # point mass pendulum: I = m * l^2 and tau_g = mgl sin(theta)
            gravity_factor = self.g / self.l
            control_factor = 1.0 / (self.m * self.l**2)
        elif self.mass_type == 'uniform':
            # uniform rod pendulum: I = (1/3) * m * l^2 and Tau_g = mg (l/2) sin(theta)
            gravity_factor = (3.0 / 2.0) * self.g / self.l
            control_factor = 3.0 / (self.m * self.l**2)
        
        A = np.array([[0., 1.],
                      [gravity_factor, -self.b * control_factor]])

        B = np.array([[0.], [control_factor]])

        return A, B

    def define_cost_matrices(self):
        '''
        Defines cost matrices for the LQR cost objective.
        Returns:
            Q, R: Cost matrices for state and input.
        '''
        # penalties for lqr
        if self.mass_type == 'uniform':
            R = 0.01 * np.eye(1)
            Q = np.diag([1., 0.1])
        if self.mass_type == 'point':
            R = 0.1 * np.eye(1)
            Q = np.diag([1., 1.])
        return Q, R    
 
    def policy(self, observation):
        # observation is by default a vector of length 3 (x, y, theta_dot)
        # convert x and y to theta with inverse tangent
        # check if batch dimension is present
        if len(observation.shape) == 1:
            batch_dim = False
        elif len(observation.shape) == 2:
            batch_dim = True
            observation = observation[0]
            
        # shape = (1,) after multiplying 1x2 and 2x
        # u = -Kx
        u = PendulumLQR.get_input(self.K, observation)
        if batch_dim:
            u = u[None, :]
        return u

