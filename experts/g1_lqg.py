from numpy.testing import measure
import mujoco
import numpy as np
from scipy import linalg
from .lqr import LQRPolicy
from stable_baselines3.common.policies import ActorCriticPolicy
from scipy.spatial.transform import Rotation

class G1LQG(LQRPolicy):
    '''
    LQR For Unitree G1 2-Leg Balance Environment.
    '''
    def __init__(self, env):
        self.env = env.unwrapped
        self.model = self.env.model
        self.data = self.env.data
        # number of DoFs
        self.nv = self.model.nv
        self.ns = self.model.nsensordata
        # number of actuators (23)
        self.nu = self.model.nu 
        # very important this matches the simulation dt
        self.dt = 0.02
        self.A, self.B, self.C, self.D = self.define_state_space_matrices()
        self.Q, self.R, self.Qo, self.Ro = self.define_cost_matrices()
        self.K = G1LQG.lqr(self.A, self.B, self.Q, self.R)

        self.L = G1LQG.lqr_obsv(self.A, self.C, self.Qo, self.Ro)
        print(self.K.shape, self.L.shape)
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

    @staticmethod
    def lqr_obsv(A, C, Q, R):
        '''
        Get the injection matrix L
        '''
        P = linalg.solve_discrete_are(A.T, C.T, Q, R)
        L = P @ C.T @ np.linalg.inv(R + C @ P @ C.T)
        return L

    def reset(self, measurement):
        '''
        Resets estimation of current state (only used for observer model).
        Measurement (53, ): base_quat, base_angular_velocity, joint_pos, joint_vel
        '''  
        base_quat = measurement[:4]
        base_angular_velocity = measurement[4:7]
        joint_pos = measurement[7:30]
        joint_vel = measurement[30:53]

        self.xhat = np.zeros(58)
        
        # Assume base position is at reference position on reset
        base_pos_ref = self.qpos0[:3]
        qpos_measured = np.concatenate([base_pos_ref, base_quat, joint_pos])
        
        # Compute deviation from reference (3-axis quaternion deviation + joint deviations)
        dq = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, dq, 1, self.qpos0, qpos_measured)
        
        # Set position deviations (first 29 elements: 3 quaternion deviation + 23 joint deviations)
        self.xhat[:29] = dq
        
        # Set velocity deviations (last 29 elements: 3 angular velocity + 23 joint velocities)
        self.xhat[29:32] = base_angular_velocity  # angular velocity deviation (assuming reference is 0)
        self.xhat[32:55] = joint_vel  # joint velocity deviation (assuming reference is 0)

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
        C = np.zeros((self.ns, 2*self.nv))
        D = np.zeros((self.ns, self.nu))
        epsilon = 1e-6
        flg_centered = True
        # uses finite difference to get state space model.
        mujoco.mjd_transitionFD(self.model, self.data, epsilon, flg_centered, A, B, C, D)

        # C and D matrices contain unrealistic sensors in XML file that the real robot doesnt have.
        # filter out unrealistic sensors. 
        sensor_names = []
        sensor_idx = []

        idx = 0
        for i in range(self.model.nsensor):
            sensor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if not ('torque' in sensor_name or 'frame' in sensor_name or 'acc' in sensor_name) :
                sensor_names.append(sensor_name)
                if 'imu_quat' in sensor_name:
                    # imu data has 4 elements (only works because all prev sensors are size 1)
                    sensor_idx.append(idx)
                    sensor_idx.append(idx + 1)
                    sensor_idx.append(idx + 2)
                    sensor_idx.append(idx + 3)
                    idx += 3
                elif 'imu_gyro' in sensor_name:
                    # angular velocity is 3 elements
                    sensor_idx.append(idx)
                    sensor_idx.append(idx + 1)
                    sensor_idx.append(idx + 2)
                    idx += 2
                else:
                    sensor_idx.append(idx)
                
            idx += 1


        # Filter C matrix
        C = C[sensor_idx, :]
        D = D[sensor_idx, :]

        return A, B, C, D

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


        num_measurements = 53
        sigma_quat = 1e-3
        sigma_gyro = 1e-3
        sigma_other = 1e-5
        # model uncertainty
        Q_obs = 1e-5 * np.eye(2*self.nv)  
        # measurement noise
        R_obs = sigma_other**2 * np.eye(num_measurements)  
        # quaternion
        R_obs[-7:-3, -7:-3] = np.eye(4) * sigma_quat**2
        # angular velocity measurement
        R_obs[-3:, -3:] = np.eye(3) * sigma_gyro**2

        return Q, R, Q_obs, R_obs

    def policy(self, observation):
        # check if batch dimension is present
        if len(observation.shape) == 1:
            batch_dim = False
        elif len(observation.shape) == 2:
            batch_dim = True
            observation = observation[0]


        # obs: [quat (4), orientation (3), angular_velocity(3), gravity_orientation(3), relative_joint_pos(23), joint_vel(23)]
        quat = observation[:4]
        # orientation = observation[4:7]
        angular_velocity = observation[7:10]
        # relative_gravity_orientation = observation[10:13] - np.array([0, 0, -1])
        relative_joint_pos = observation[13:36] 
        joint_vel = observation[36:59]
        joint_pos = relative_joint_pos + self.qpos0[7:]

        # qpos = np.concatenate([quat, joint_pos])
        # vel = np.concatenate([angular_velocity, joint_vel])

        # Reorder measurement to match C matrix: [joint_pos, joint_vel, quat, angular_velocity]
        measurement = np.concatenate([joint_pos, joint_vel, quat, angular_velocity])
        y = measurement

        # control law 
        u = - self.K @ self.xhat + self.ctrl0

        # Observer update
        self.xhat = self.xhat + self.dt*(self.A @ self.xhat + self.B @ (u - self.ctrl0) - self.L @ (self.C @ self.xhat - y))

        if self.env._normalized_actions:
            # normalize action from -1 to 1
            u = 2 * (u - self.env._original_action_space.low) / (self.env._original_action_space.high - self.env._original_action_space.low) - 1.

        if batch_dim:
            u = u[None, :]

        return u