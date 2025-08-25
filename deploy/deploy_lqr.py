import sys, os
import numpy as np
import time
from scipy.spatial.transform import Rotation
import gymnasium as gym

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from common.command_helper import create_damping_cmd
from config import Config
from deploy_real import Controller
from common.rotation_helper import get_gravity_orientation, transform_imu_data
from common.remote_controller import KeyMap


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experts.g1_lqr import G1LQR

class LQRController(Controller):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize MuJoCo environment using gym.make
        self.env = gym.make('G1Balance-v0')
        
        # Initialize LQR policy with the environment
        self.lqr_policy = G1LQR(self.env)
        
        # Define torque limits based on MuJoCo environment (with safety factor)
        safety_factor = 1.0
        
        self.torque_limits = np.array([88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50,
                                       88,
                                       25, 25, 25, 25, 25,
                                       25, 25, 25, 25, 25], dtype=np.float32)
        
        self.torque_limits *= safety_factor
        self.action_low = -self.torque_limits
        self.action_high = self.torque_limits

    def get_state(self):
        '''
        Gets the observation of size 62
        '''
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

    def run(self):
        self.counter += 1
        all_joint2motor_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        obs = self.get_state()
        
        self.action = self.lqr_policy.policy(obs)

        # Clip torques to safety limits
        desired_torques = np.clip(self.action, self.action_low, self.action_high)

        # Build low cmd
        # for i in range(len(all_joint2motor_idx)):
        #     motor_idx = all_joint2motor_idx[i]
        #     self.low_cmd.motor_cmd[motor_idx].q = 0
        #     self.low_cmd.motor_cmd[motor_idx].qd = 0
        #     self.low_cmd.motor_cmd[motor_idx].kp = 0
        #     self.low_cmd.motor_cmd[motor_idx].kd = 0
        #     # action is 23 torques (already in expected order)
        #     self.low_cmd.motor_cmd[motor_idx].tau = desired_torques[i]

        # send the command
        # self.send_cmd(self.low_cmd)
        # time.sleep(self.config.control_dt)

        


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

    controller = LQRController(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()


    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")