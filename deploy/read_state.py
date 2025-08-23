import os
import numpy as np
import time

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
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion
        # self.low_state.imu_state
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.ly
        self.cmd[1] = self.remote_controller.lx * -1
        self.cmd[2] = self.remote_controller.rx * -1

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

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