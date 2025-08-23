import mujoco
import numpy as np
from scipy import linalg
from .lqr import LQRPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from scipy.spatial.transform import Rotation

class G1LQR(LQRPolicy):
    '''
    LQR For Unitree G1 2-Leg Balance Environment.
    '''
    def __init__(self, env):
        self.env = env.unwrapped
        self.model = self.env.model
        self.data = self.env.data
        # number of DoFs
        self.nv = self.model.nv
        # number of actuators (23)
        self.nu = self.model.nu 
        self.A, self.B = self.define_state_space_matrices()
        self.Q, self.R = self.define_cost_matrices()
        self.K = G1LQR.lqr(self.A, self.B, self.Q, self.R)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        lr_schedule = lambda _: 0.
        ActorCriticPolicy.__init__(self, self.observation_space, self.action_space, lr_schedule=lr_schedule)

    @staticmethod
    def lqr(A, B, Q, R):
        '''
        Solve for optimal LQR feedback gain.
        '''
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        # remove root position and velocity indices (12 elements)
        K_filtered = K[:, 6:29]
        K_filtered = np.concatenate([K_filtered, K[:, 35:58]], axis=1)
        return K_filtered

    def define_state_space_matrices(self):
        '''
        Defines state space matrices for the system x_dot = Ax + Bu.
        Returns:
            A, B: State space matrices.
        '''
        # reset to target stadning positon position, find control setpoint using inverse dynamics, calculate state space matrices
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.data.qacc = 0
        # position setpoint
        qpos0 = self.data.qpos.copy()
        self.qpos0 = qpos0
        # inverse dynamics function takes the acceleration as input and computes the forces required to create the acceleration
        mujoco.mj_inverse(self.model, self.data)
        # forces required
        qfrc0 = self.data.qfrc_inverse.copy()
        # divide by the actuation moment arm matrix, i.e. multiply by its pseudo-inverse to find actuation force
        actuator_moment = np.zeros((self.model.nu, self.model.nv))
        mujoco.mju_sparse2dense(
            actuator_moment,
            self.data.actuator_moment.reshape(-1),
            self.data.moment_rownnz,
            self.data.moment_rowadr,
            self.data.moment_colind.reshape(-1),
        )
        ctrl0 = np.atleast_2d(qfrc0) @ np.linalg.pinv(actuator_moment)
        # control setpoint
        ctrl0 = ctrl0.flatten()
        self.ctrl0 = ctrl0

        # Set the state and controls to their setpoints.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos = qpos0
        # already initialized to 0 but set to 0 for completeness
        self.data.qvel = np.zeros_like(self.data.qvel)
        self.data.ctrl = ctrl0
        # Allocate the A and B matrices, compute them.
        A = np.zeros((2*self.nv, 2*self.nv))
        B = np.zeros((2*self.nv, self.nu))
        epsilon = 1e-6
        flg_centered = True
        mujoco.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B, None, None)

        return A, B

    def define_cost_matrices(self):
        '''
        Defines cost matrices for the LQR cost objective.
        Returns:
            Q, R: Cost matrices for state and input.
        '''

        # Get all joint names.
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]

        # Get indices into relevant sets of joints.
        root_dofs = range(6)
        body_dofs = range(6, self.nv)

        waist_dofs = [
            self.model.joint(name).dofadr[0]
            for name in joint_names
            if 'waist' in name
        ]
        right_leg_dofs = [
            self.model.joint(name).dofadr[0]
            for name in joint_names
            if 'right' in name
            and ('hip' in name or 'knee' in name or 'ankle' in name)
        ]
        left_leg_dofs = [
            self.model.joint(name).dofadr[0]
            for name in joint_names
            if 'left' in name
            and ('hip' in name or 'knee' in name or 'ankle' in name)
        ]
        balance_dofs = left_leg_dofs + waist_dofs

        balance_dofs = balance_dofs + right_leg_dofs
            
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)

        # penalties for lqr

        BALANCE_COST        = 1000  # Balancing.
        BALANCE_JOINT_COST  = 3    # Joints required for balancing.
        OTHER_JOINT_COST    = 0.3    # Other joints.
        R = np.eye(self.nu)
        
        # Q matrix for joint-only root + joint
        Q = 10 * np.eye(58)
        Q[range(self.nu), range(self.nu)] = BALANCE_COST  # First 23 diagonal elements (joint positions)
        Q[root_dofs, root_dofs] = 0
        # Q[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        # Q[other_dofs, other_dofs] *= OTHER_JOINT_COST

        return Q, R    
 
    def quaternion_difference(self, q1, q2):
        """
        Compute the difference between two quaternions in the tangent space.
        Returns a 3D vector representing the angular difference.
        """
        # Convert to scipy format [x, y, z, w]
        q1_scipy = np.array([q1[1], q1[2], q1[3], q1[0]])  # [x, y, z, w]
        q2_scipy = np.array([q2[1], q2[2], q2[3], q2[0]])  # [x, y, z, w]
        
        # Compute relative rotation
        r1 = Rotation.from_quat(q1_scipy)
        r2 = Rotation.from_quat(q2_scipy)
        relative_rot = r2 * r1.inv()
        
        # Convert to axis-angle representation (3D)
        axis_angle = relative_rot.as_rotvec()  # This is 3D!
        
        return axis_angle

    def policy(self, observation):
        # check if batch dimension is present
        if len(observation.shape) == 1:
            batch_dim = False
        elif len(observation.shape) == 2:
            batch_dim = True
            observation = observation[0]

        
        qpos = observation[:30]
        # make joints 
        qpos[6:] = qpos[6:] + self.qpos0[6:]
        qvel = observation[30:59] 
        
        # Compute state difference from equilibrium
        dq = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1, self.qpos0, qpos)
        
        dx = np.concatenate([dq[6:], qvel[6:]])

        # LQR control law: u = ctrl0 - Kx
        du = - self.K @ dx
        u = self.ctrl0 + du 
        if self.env._normalized_actions:
            # normalize action from -1 to 1
            u = 2 * (u - self.env._original_action_space.low) / (self.env._original_action_space.high - self.env._original_action_space.low) - 1.

        if batch_dim:
            u = u[None, :]


        return u

