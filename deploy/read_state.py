#!/usr/bin/env python3.10
"""lqr_balance.py – G1 robot observation reader based on deploy_real.py

This script reads robot state and prints observations to terminal.
No policy, no control - just data reading for testing/debugging.

Usage:
    python lqr_balance.py enp68s0f1
"""

import numpy as np
import time
import argparse

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.utils.crc import CRC


class G1Observer:
    """G1 robot observer that reads and prints robot state."""
    
    def __init__(self, net_interface: str):
        self.net_interface = net_interface
        self.control_dt = 0.01  # 100Hz control rate
        
        # Initialize DDS communication
        ChannelFactoryInitialize(0, net_interface)
        
        # Initialize state and command messages
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        
        # Initialize publishers and subscribers
        self.lowcmd_publisher_ = ChannelPublisher("rt/arm_sdk", LowCmdHG)
        self.lowcmd_publisher_.Init()
        
        self.lowstate_subscriber = ChannelSubscriber("rt/robot_state", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        # Wait for connection
        self.wait_for_low_state()
        
        # Initialize command message
        self.init_cmd()
        
        print(f"Successfully connected to G1 robot on {net_interface}")

    def LowStateHgHandler(self, msg: LowStateHG):
        """Handle incoming robot state messages."""
        self.low_state = msg

    def init_cmd(self):
        """Initialize command message with default settings."""
        # Set default motor parameters
        for mc in self.low_cmd.motor_cmd:
            mc.mode = 0  # Position control mode
            mc.kp = 40.0
            mc.kd = 1.0
            mc.q = 0.0
            mc.qd = 0.0
            mc.tau = 0.0
        
        # Enable arm control
        if len(self.low_cmd.motor_cmd) > 29:
            self.low_cmd.motor_cmd[29].q = 1.0

    def send_cmd(self, cmd):
        """Send command to robot."""
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        """Wait for robot state connection."""
        while self.low_state.tick == 0:
            time.sleep(self.control_dt)
        print("Successfully connected to the robot.")

    def get_robot_observation(self):
        """Get current robot observation in the format expected by LQR."""
        # Extract joint positions and velocities
        joint_positions = []
        joint_velocities = []
        
        # Get all motor states (should be 30 motors for G1)
        for i, motor_state in enumerate(self.low_state.motor_state):
            joint_positions.append(float(motor_state.q))
            joint_velocities.append(float(motor_state.dq))
        
        # Get IMU data
        imu_quat = self.low_state.imu_state.quaternion  # [w, x, y, z]
        imu_gyro = np.array([
            self.low_state.imu_state.gyroscope[0],
            self.low_state.imu_state.gyroscope[1], 
            self.low_state.imu_state.gyroscope[2]
        ])
        imu_accel = np.array([
            self.low_state.imu_state.accelerometer[0],
            self.low_state.imu_state.accelerometer[1],
            self.low_state.imu_state.accelerometer[2]
        ])
        
        # Create observation array: [base_pos(7), base_vel(6), joint_pos(23), joint_vel(23)]
        # For now, we'll use zeros for base position since we don't have position data
        observation = np.zeros(59)
        
        # Base position (zeros for now - would need external positioning system)
        observation[0:3] = [0.0, 0.0, 0.79127]  # Default height
        
        # Base orientation from IMU quaternion
        observation[3:7] = imu_quat  # [w, x, y, z]
        
        # Base linear velocity (zeros for now)
        observation[7:10] = [0.0, 0.0, 0.0]
        
        # Base angular velocity from IMU
        observation[10:13] = imu_gyro
        
        # Joint positions (indices 13-35)
        for i in range(min(23, len(joint_positions))):
            observation[13 + i] = joint_positions[i]
        
        # Joint velocities (indices 36-58)
        for i in range(min(23, len(joint_velocities))):
            observation[36 + i] = joint_velocities[i]
        
        return observation, {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'imu_quat': imu_quat,
            'imu_gyro': imu_gyro,
            'imu_accel': imu_accel,
            'timestamp': time.time()
        }

    def print_observation(self, observation, raw_data):
        """Print observation data in a readable format."""
        print("\n" + "="*80)
        print(f"G1 ROBOT OBSERVATION - {time.strftime('%H:%M:%S')}")
        print("="*80)
        
        # Print base state
        print("BASE STATE:")
        print(f"  Position (x,y,z): [{observation[0]:.4f}, {observation[1]:.4f}, {observation[2]:.4f}]")
        print(f"  Orientation (w,x,y,z): [{observation[3]:.4f}, {observation[4]:.4f}, {observation[5]:.4f}, {observation[6]:.4f}]")
        print(f"  Linear velocity: [{observation[7]:.4f}, {observation[8]:.4f}, {observation[9]:.4f}]")
        print(f"  Angular velocity: [{observation[10]:.4f}, {observation[11]:.4f}, {observation[12]:.4f}]")
        
        # Print IMU data
        print(f"\nIMU DATA:")
        print(f"  Accelerometer: [{raw_data['imu_accel'][0]:.4f}, {raw_data['imu_accel'][1]:.4f}, {raw_data['imu_accel'][2]:.4f}]")
        
        # Print joint positions
        print(f"\nJOINT POSITIONS:")
        joint_names = [
            "L_hip_pitch", "L_hip_roll", "L_hip_yaw", "L_knee", "L_ankle_pitch", "L_ankle_roll",
            "R_hip_pitch", "R_hip_roll", "R_hip_yaw", "R_knee", "R_ankle_pitch", "R_ankle_roll",
            "waist_yaw",
            "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow", "L_wrist_roll", "L_wrist_pitch", "L_wrist_yaw",
            "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow", "R_wrist_roll", "R_wrist_pitch", "R_wrist_yaw"
        ]
        
        for i, name in enumerate(joint_names):
            if i < len(raw_data['joint_positions']):
                pos = raw_data['joint_positions'][i]
                vel = raw_data['joint_velocities'][i] if i < len(raw_data['joint_velocities']) else 0.0
                print(f"  {name:15s}: pos={pos:8.4f} rad, vel={vel:8.4f} rad/s")
        
        # Print statistics
        print(f"\nSTATISTICS:")
        print(f"  Joint position range: [{observation[13:36].min():.4f}, {observation[13:36].max():.4f}]")
        print(f"  Joint velocity range: [{observation[36:59].min():.4f}, {observation[36:59].max():.4f}]")
        print(f"  Observation norm: {np.linalg.norm(observation):.4f}")
        print(f"  Number of joints: {len(raw_data['joint_positions'])}")
        print(f"  Timestamp: {raw_data['timestamp']:.3f}")
        print("="*80)

    def run_observation_loop(self, max_steps=1000, print_rate=10):
        """Run the observation reading loop."""
        print(f"Starting observation loop at {1/self.control_dt:.1f} Hz")
        print(f"Printing every {print_rate} steps")
        print("Press Ctrl+C to stop")
        
        step_count = 0
        
        try:
            while step_count < max_steps:
                start_time = time.time()
                
                # Get observation
                observation, raw_data = self.get_robot_observation()
                
                # Print observation at specified rate
                if step_count % print_rate == 0:
                    self.print_observation(observation, raw_data)
                
                step_count += 1
                
                # Rate limiting
                elapsed = time.time() - start_time
                if elapsed < self.control_dt:
                    time.sleep(self.control_dt - elapsed)
                else:
                    print(f"Warning: loop took {elapsed*1000:.1f}ms, target was {self.control_dt*1000:.1f}ms")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in observation loop: {e}")
        finally:
            print(f"Observation loop ended after {step_count} steps")


def main():
    parser = argparse.ArgumentParser(description="G1 Reader")
    parser.add_argument("net", type=str, help="network interface (e.g., enp68s0f1)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum observation steps")
    parser.add_argument("--print-rate", type=int, default=10, help="Print every N steps")
    
    args = parser.parse_args()
    
    try:
        # Initialize observer
        observer = G1Observer(args.net) 
        # Run observation loop
        observer.run_observation_loop(args.max_steps, args.print_rate)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
