import os
import numpy as np
import time
from scipy.spatial.transform import Rotation

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from common.command_helper import create_damping_cmd
from config import Config
from deploy_real import Controller
from common.rotation_helper import get_gravity_orientation, transform_imu_data

class StateReader(Controller):
    def __init__(self, config):
        super().__init__(config)


    def get_state(self):
        '''
        Gets the observation of size 47

        Current format:
            angular velocity (3,)
            gravity (3,)
            command_input (lx, ly, rx) for remote controller commands (3,) 
            joint_positions (12, )
            joiint_velicities (12,)
            previous_actions (12, )
            phase_info (2, )

        '''
        self.counter += 1
        # Get the current joint position and velocity
        all_joint2motor_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        for i in range(len(all_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[all_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[all_joint2motor_idx[i]].dq



        # create observation
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion

        # imu acceleration reads [0, 0, 9.81] at perfect rest in standing mode
        gravity_world = np.array([0., 0., -9.81], dtype=np.float32)  # Gravity in world frame
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        R = Rotation.from_quat(quat_xyzw)
        acc_body = self.low_state.imu_state.accelerometer
        acc_world = R.apply(acc_body)
        acc = acc_world + gravity_world
        
        orientation = self.low_state.imu_state.rpy
        # self.low_state.imu_state
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32) * self.config.ang_vel_scale
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale

        self.obs[:3] = acc
        self.obs[3:7] = quat
        self.obs[7:10] = orientation
        self.obs[10:13] = ang_vel
        self.obs[13:16] = gravity_orientation
        self.obs[16:39] = qj_obs
        self.obs[39:62] = dqj_obs

        return self.obs


        


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1_lqr.yaml")
    args = parser.parse_args()

    # Load config
    cur_dir = os.path.dirname(__file__)
    config_path = f"{cur_dir}/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    reader = StateReader(config)

    # Enter the zero torque state, press the start key to continue executing
    reader.zero_torque_state()

    # Move to the default position
    reader.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    reader.default_pos_state()


    obs = reader.get_state()

    print(obs.shape)
    print(obs)

    create_damping_cmd(reader.low_cmd)
    reader.send_cmd(reader.low_cmd)
    print("Exit")