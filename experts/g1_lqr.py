from numpy.testing import measure
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
        self.dt = 0.02
        self.A, self.B = self.define_state_space_matrices()
        self.Q, self.R = self.define_cost_matrices()
        self.K = G1LQR.lqr(self.A, self.B, self.Q, self.R)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        lr_schedule = lambda _: 0.
        ActorCriticPolicy.__init__(self, self.observation_space, self.action_space, lr_schedule=lr_schedule)

        # sanity checks
        self.checkControllable(self.A, self.B)

        self.env.unwrapped.set_controller(self)

    @staticmethod
    def lqr(A, B, Q, R):
        '''
        Solve for optimal LQR feedback gain.
        '''
        P = linalg.solve_discrete_are(A, B, Q, R)
        K = linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        return K

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
        # uses finite difference to get state space model.
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
        
        Q = 10 * np.eye(58)
        Q[range(self.nv), range(self.nv)] = BALANCE_COST
        Q[root_dofs, root_dofs] = 0
        Q[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
        Q[other_dofs, other_dofs] *= OTHER_JOINT_COST


        return Q, R

    def reset(self, measurement):
        '''
        Resets the integration state when environment resets.
        Measurement (53, ): base_quat, base_angular_velocity, joint_pos, joint_vel
        # '''  
        # base_quat = measurement[:4]
        # base_angular_velocity = measurement[4:7]
        # joint_pos = measurement[7:30]
        # joint_vel = measurement[30:53]

        # Reset base position and velocity for integration
        # Assume base position is at reference position on reset
        self.base_pos = self.qpos0[:3]
        self.base_vel = np.zeros(3)

    def policy(self, observation):
        # check if batch dimension is present
        if len(observation.shape) == 1:
            batch_dim = False
        elif len(observation.shape) == 2:
            batch_dim = True
            observation = observation[0]


        # obs: [quat (4), orientation (3), angular_velocity(3), gravity_orientation(3), base_accel, relative_joint_pos(23), joint_vel(23)]
        base_quat = observation[3:7]
        base_orientation = observation[7:10]
        base_angular_velocity = observation[10:13]
        relative_gravity_orientation = observation[10:13] - np.array([0, 0, -1])
        base_acc = observation[13:16]
        relative_joint_pos = observation[16:39] 
        joint_vel = observation[39:62]
        joint_pos = relative_joint_pos + self.qpos0[7:]

        # quat_xyzw = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        # R = Rotation.from_quat(quat_xyzw)
        # base_acc_world = R.apply(base_acc)
        
        # integration
        self.base_vel = self.base_vel + self.dt * base_acc
        self.base_pos = self.base_pos + self.dt * self.base_vel
        
        qpos = np.concatenate([self.base_pos, base_quat, joint_pos])
        qvel = np.concatenate([self.base_vel, base_angular_velocity, joint_vel])
        dq = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1, self.qpos0, qpos)
        dx = np.hstack((dq, qvel)).T
        # control law 
        u = - self.K @ dx + self.ctrl0

        if self.env._normalized_actions:
            # normalize action from -1 to 1
            u = 2 * (u - self.env._original_action_space.low) / (self.env._original_action_space.high - self.env._original_action_space.low) - 1.

        if batch_dim:
            u = u[None, :]

        return u