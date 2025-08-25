import numpy as np
import os

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

def quat_to_euler(quat):
    """
    Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
    Uses ZYX (yaw-pitch-roll) convention.
    """
    w, x, y, z = quat
    
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2*(w*y - z*x))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    
    return np.array([roll, pitch, yaw])

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def cosine_similarity_angles(angles, target_angles=None):
    """
    Compute cosine similarity for angles, where small angles give higher rewards.
    
    Args:
        angles: Array of angles in radians
        target_angles: Target angles (defaults to zeros)
    
    Returns:
        Array of cosine similarities, where 1 = perfect alignment, -1 = opposite
    """
    if target_angles is None:
        target_angles = np.zeros_like(angles)
    
    # Convert angles to unit vectors on the unit circle
    angle_vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    target_vectors = np.column_stack([np.cos(target_angles), np.sin(target_angles)])
    
    # Compute cosine similarity
    similarities = np.sum(angle_vectors * target_vectors, axis=1)
    
    return similarities

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance":2.5,
    "lookat": np.array((0.0, 0.0, 0.8)),
    "elevation": -20.0,
    "azimuth": -130.0,
}


class G1BalanceEnv(MujocoEnv, utils.EzPickle):
    """
    Unitree G1 balance environment.
    The robot has 23 degrees of freedom.
    The goal is to maintain balance while standing on 2 legs. External forces are applied to perturb the robot slightly.

    This gym environment runs the controller (and display) at 100 hz (1 / 0.01 s) and runs mujoco internally at dt=0.002 (500 hz).

    ## Action Space

    The action space is continuous and size 23. Each action represents a torque applied to the motor.
    The action space internally stores torque low and high commands from the xml file.

    | Num | Action                                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    | --- | ---------------------------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the left hip pitch joint                                         | -88         | 88          | left_hip_pitch_joint             | hinge | torque (N m) |
    | 1   | Torque applied on the left hip roll joint                                          | -88         | 88          | left_hip_roll_joint              | hinge | torque (N m) |
    | 2   | Torque applied on the left hip yaw joint                                           | -88         | 88          | left_hip_yaw_joint               | hinge | torque (N m) |
    | 3   | Torque applied on the left knee joint                                              | -139        | 139         | left_knee_joint                  | hinge | torque (N m) |
    | 4   | Torque applied on the left ankle pitch joint                                       | -50         | 50          | left_ankle_pitch_joint           | hinge | torque (N m) |
    | 5   | Torque applied on the left ankle roll joint                                        | -50         | 50          | left_ankle_roll_joint            | hinge | torque (N m) |
    | 6   | Torque applied on the right hip pitch joint                                        | -88         | 88          | right_hip_pitch_joint            | hinge | torque (N m) |
    | 7   | Torque applied on the right hip roll joint                                         | -88         | 88          | right_hip_roll_joint             | hinge | torque (N m) |
    | 8   | Torque applied on the right hip yaw joint                                          | -88         | 88          | right_hip_yaw_joint              | hinge | torque (N m) |
    | 9   | Torque applied on the right knee joint                                             | -139        | 139         | right_knee_joint                 | hinge | torque (N m) |
    | 10  | Torque applied on the right ankle pitch joint                                      | -50         | 50          | right_ankle_pitch_joint          | hinge | torque (N m) |
    | 11  | Torque applied on the right ankle roll joint                                       | -50         | 50          | right_ankle_roll_joint           | hinge | torque (N m) |
    | 12  | Torque applied on the waist yaw joint                                              | -88         | 88          | waist_yaw_joint                  | hinge | torque (N m) |
    | 13  | Torque applied on the left shoulder pitch joint                                    | -25         | 25          | left_shoulder_pitch_joint        | hinge | torque (N m) |
    | 14  | Torque applied on the left shoulder roll joint                                     | -25         | 25          | left_shoulder_roll_joint         | hinge | torque (N m) |
    | 15  | Torque applied on the left shoulder yaw joint                                      | -25         | 25          | left_shoulder_yaw_joint          | hinge | torque (N m) |
    | 16  | Torque applied on the left elbow joint                                             | -25         | 25          | left_elbow_joint                 | hinge | torque (N m) |
    | 17  | Torque applied on the left wrist roll joint                                        | -25         | 25          | left_wrist_roll_joint            | hinge | torque (N m) |
    | 18  | Torque applied on the right shoulder pitch joint                                   | -25         | 25          | right_shoulder_pitch_joint       | hinge | torque (N m) |
    | 19  | Torque applied on the right shoulder roll joint                                    | -25         | 25          | right_shoulder_roll_joint        | hinge | torque (N m) |
    | 20  | Torque applied on the right shoulder yaw joint                                     | -25         | 25          | right_shoulder_yaw_joint         | hinge | torque (N m) |
    | 21  | Torque applied on the right elbow joint                                            | -25         | 25          | right_elbow_joint                | hinge | torque (N m) |
    | 22  | Torque applied on the right wrist roll joint                                       | -25         | 25          | right_wrist_roll_joint           | hinge | torque (N m) |

    ## Observation Space

    The observation space consists of the following parts (in order):

    - *base_quaternion (4 elements):* The quaternion representation of the robot's base orientation.
    - *base_orientation (3 elements):* The Euler angles (roll, pitch, yaw) of the robot's base orientation.
    - *projected_gravity (3 elements):* Gravity vector projected in the robot's base frame.
    - *angular_velocity (3 elements):* The angular velocity of the robot's base (pelvis).
    - *joint_pos (23 elements):* The relative joint positions (deviation from default standing pose).
    - *joint_vel (23 elements):* The joint velocities.
    - *previous_actions (23 elements, optional):* Previous action values for action continuity (included if `include_previous_actions=True`).

    The observation space size depends on the flags:
    - Base: 59 elements (base_quaternion + base_orientation + projected_gravity + angular_velocity + joint_pos + joint_vel)
    - With previous actions: 82 elements (59 + 23)

    | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
    | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
    | 0   | w-component of base quaternion                                                                                    | -Inf | Inf | floating_base_joint              | free  | quaternion                  |
    | 1   | x-component of base quaternion                                                                                    | -Inf | Inf | floating_base_joint              | free  | quaternion                  |
    | 2   | y-component of base quaternion                                                                                    | -Inf | Inf | floating_base_joint              | free  | quaternion                  |
    | 3   | z-component of base quaternion                                                                                    | -Inf | Inf | floating_base_joint              | free  | quaternion                  |
    | 4   | roll angle of the base (pelvis)                                                                                   | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 5   | pitch angle of the base (pelvis)                                                                                  | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 6   | yaw angle of the base (pelvis)                                                                                    | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 7   | x-component of projected gravity                                                                                   | -Inf | Inf | -                              | -     | unit vector                |
    | 8   | y-component of projected gravity                                                                                   | -Inf | Inf | -                              | -     | unit vector                |
    | 9   | z-component of projected gravity                                                                                   | -Inf | Inf | -                              | -     | unit vector                |
    | 10  | x-coordinate angular velocity of the pelvis (center)                                                             | -Inf | Inf | floating_base_joint              | free  | angular velocity (rad/s)   |
    | 11  | y-coordinate angular velocity of the pelvis (center)                                                             | -Inf | Inf | floating_base_joint              | free  | angular velocity (rad/s)   |
    | 12  | z-coordinate angular velocity of the pelvis (center)                                                             | -Inf | Inf | floating_base_joint              | free  | angular velocity (rad/s)   |
    | 13  | angle of the left hip pitch joint                                                                                | -Inf | Inf | left_hip_pitch_joint             | hinge | angle (rad)                |
    | 14  | angle of the left hip roll joint                                                                                 | -Inf | Inf | left_hip_roll_joint              | hinge | angle (rad)                |
    | 15  | angle of the left hip yaw joint                                                                                  | -Inf | Inf | left_hip_yaw_joint               | hinge | angle (rad)                |
    | 16  | angle of the left knee joint                                                                                     | -Inf | Inf | left_knee_joint                  | hinge | angle (rad)                |
    | 17  | angle of the left ankle pitch joint                                                                              | -Inf | Inf | left_ankle_pitch_joint           | hinge | angle (rad)                |
    | 18  | angle of the left ankle roll joint                                                                               | -Inf | Inf | left_ankle_roll_joint            | hinge | angle (rad)                |
    | 19  | angle of the right hip pitch joint                                                                               | -Inf | Inf | right_hip_pitch_joint            | hinge | angle (rad)                |
    | 20  | angle of the right hip roll joint                                                                                | -Inf | Inf | right_hip_roll_joint             | hinge | angle (rad)                |
    | 21  | angle of the right hip yaw joint                                                                                 | -Inf | Inf | right_hip_yaw_joint              | hinge | angle (rad)                |
    | 22  | angle of the right knee joint                                                                                    | -Inf | Inf | right_knee_joint                 | hinge | angle (rad)                |
    | 23  | angle of the right ankle pitch joint                                                                             | -Inf | Inf | right_ankle_pitch_joint          | hinge | angle (rad)                |
    | 24  | angle of the right ankle roll joint                                                                              | -Inf | Inf | right_ankle_roll_joint           | hinge | angle (rad)                |
    | 25  | angle of the waist yaw joint                                                                                     | -Inf | Inf | waist_yaw_joint                  | hinge | angle (rad)                |
    | 26  | angle of the left shoulder pitch joint                                                                            | -Inf | Inf | left_shoulder_pitch_joint        | hinge | angle (rad)                |
    | 27  | angle of the left shoulder roll joint                                                                             | -Inf | Inf | left_shoulder_roll_joint         | hinge | angle (rad)                |
    | 28  | angle of the left shoulder yaw joint                                                                              | -Inf | Inf | left_shoulder_yaw_joint          | hinge | angle (rad)                |
    | 29  | angle of the left elbow joint                                                                                     | -Inf | Inf | left_elbow_joint                 | hinge | angle (rad)                |
    | 30  | angle of the left wrist roll joint                                                                                | -Inf | Inf | left_wrist_roll_joint            | hinge | angle (rad)                |
    | 31  | angle of the right shoulder pitch joint                                                                           | -Inf | Inf | right_shoulder_pitch_joint       | hinge | angle (rad)                |
    | 32  | angle of the right shoulder roll joint                                                                            | -Inf | Inf | right_shoulder_roll_joint        | hinge | angle (rad)                |
    | 33  | angle of the right shoulder yaw joint                                                                             | -Inf | Inf | right_shoulder_yaw_joint         | hinge | angle (rad)                |
    | 34  | angle of the right elbow joint                                                                                    | -Inf | Inf | right_elbow_joint                | hinge | angle (rad)                |
    | 35  | angle of the right wrist roll joint                                                                               | -Inf | Inf | right_wrist_roll_joint           | hinge | angle (rad)                |
    | 36  | angular velocity of the left hip pitch joint                                                                     | -Inf | Inf | left_hip_pitch_joint             | hinge | angular velocity (rad/s)   |
    | 37  | angular velocity of the left hip roll joint                                                                      | -Inf | Inf | left_hip_roll_joint              | hinge | angular velocity (rad/s)   |
    | 38  | angular velocity of the left hip yaw joint                                                                       | -Inf | Inf | left_hip_yaw_joint               | hinge | angular velocity (rad/s)   |
    | 39  | angular velocity of the left knee joint                                                                          | -Inf | Inf | left_knee_joint                  | hinge | angular velocity (rad/s)   |
    | 40  | angular velocity of the left ankle pitch joint                                                                   | -Inf | Inf | left_ankle_pitch_joint           | hinge | angular velocity (rad/s)   |
    | 41  | angular velocity of the left ankle roll joint                                                                    | -Inf | Inf | left_ankle_roll_joint            | hinge | angular velocity (rad/s)   |
    | 42  | angular velocity of the right hip pitch joint                                                                    | -Inf | Inf | right_hip_pitch_joint            | hinge | angular velocity (rad/s)   |
    | 43  | angular velocity of the right hip roll joint                                                                     | -Inf | Inf | right_hip_roll_joint             | hinge | angular velocity (rad/s)   |
    | 44  | angular velocity of the right hip yaw joint                                                                      | -Inf | Inf | right_hip_yaw_joint              | hinge | angular velocity (rad/s)   |
    | 45  | angular velocity of the right knee joint                                                                         | -Inf | Inf | right_knee_joint                 | hinge | angular velocity (rad/s)   |
    | 46  | angular velocity of the right ankle pitch joint                                                                  | -Inf | Inf | right_ankle_pitch_joint          | hinge | angular velocity (rad/s)   |
    | 47  | angular velocity of the right ankle roll joint                                                                   | -Inf | Inf | right_ankle_roll_joint           | hinge | angular velocity (rad/s)   |
    | 48  | angular velocity of the waist yaw joint                                                                          | -Inf | Inf | waist_yaw_joint                  | hinge | angular velocity (rad/s)   |
    | 49  | angular velocity of the left shoulder pitch joint                                                                 | -Inf | Inf | left_shoulder_pitch_joint        | hinge | angular velocity (rad/s)   |
    | 50  | angular velocity of the left shoulder roll joint                                                                  | -Inf | Inf | left_shoulder_roll_joint         | hinge | angular velocity (rad/s)   |
    | 51  | angular velocity of the left shoulder yaw joint                                                                   | -Inf | Inf | left_shoulder_yaw_joint          | hinge | angular velocity (rad/s)   |
    | 52  | angular velocity of the left elbow joint                                                                          | -Inf | Inf | left_elbow_joint                 | hinge | angular velocity (rad/s)   |
    | 53  | angular velocity of the left wrist roll joint                                                                     | -Inf | Inf | left_wrist_roll_joint            | hinge | angular velocity (rad/s)   |
    | 54  | angular velocity of the right shoulder pitch joint                                                                | -Inf | Inf | right_shoulder_pitch_joint       | hinge | angular velocity (rad/s)   |
    | 55  | angular velocity of the right shoulder roll joint                                                                 | -Inf | Inf | right_shoulder_roll_joint        | hinge | angular velocity (rad/s)   |
    | 56  | angular velocity of the right shoulder yaw joint                                                                  | -Inf | Inf | right_shoulder_yaw_joint         | hinge | angular velocity (rad/s)   |
    | 57  | angular velocity of the right elbow joint                                                                         | -Inf | Inf | right_elbow_joint                | hinge | angular velocity (rad/s)   |
    | 58  | angular velocity of the right wrist roll joint                                                                    | -Inf | Inf | right_wrist_roll_joint           | hinge | angular velocity (rad/s)   |

    The (x,y,z) coordinates are translational DOFs, while the orientations are rotational DOFs expressed as quaternions.
    One can read more about free joints in the [MuJoCo documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    ## Rewards
    The reward function includes the following components:
    
    - *Terminate penalty:* -250 for losing balance
    - *Orientation cost:* Penalty for non-upright orientation (using projected gravity)
    - *Joint position cost:* Penalty for deviating from reference joint positions using cosine similarity
    - *Joint velocity cost:* Penalty for high joint velocities
    - *Control cost:* Penalty for large control inputs
    - *Action rate cost:* Penalty for large changes between consecutive actions (smoothness)

    ## Starting State
    The robot starts in the "stand" keyframe position with some noise added for stochasticity.

    ## Episode End
    The robot is said to be unhealthy if the z-position of the pelvis is no longer contained in the
    closed interval specified by the argument `healthy_z_range`.

    ## Arguments
    | Parameter                                    | Type      | Default          | Description                                                                                                                                                               |
    | -------------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `frame_skip`                                 | **int**   | `10`              | how many MuJoCo physics steps happen between each gym environment step. By default, step() runs at 50 hz, so MuJoCo internally updates at 500 hz.                        |
    | `normalized_actions`                           | **bool**  | `True`           | If true, normalize actions to [-1, 1] range for better RL training stability                                                                                              |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        frame_skip=1,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        normalized_actions=True,
        include_previous_actions=False,
        action_smoothness=False,
        render_mode=None,
    ):
        utils.EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            normalized_actions,
            include_previous_actions,
            action_smoothness,
            render_mode,
        )
        # healthy roll and pitch range
        # roll and pitch can't exceed 78.75 degrees
        self._healthy_rp_range = 7 * np.pi / 16

        self._position_weight = np.ones(23)
        # Left leg: hip pitch(0), hip roll(1), hip yaw(2), knee(3), ankle pitch(4), ankle roll(5)
        # Right leg: hip pitch(6), hip roll(7), hip yaw(8), knee(9), ankle pitch(10), ankle roll(11)
        leg_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 leg joints
        waist_index = 12
        # Left arm: shoulder pitch(13), shoulder roll(14), shoulder yaw(15), elbow(16), wrist roll(17)
        # Right arm: shoulder pitch(18), shoulder roll(19), shoulder yaw(20), elbow(21), wrist roll(22)
        arm_indices = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

        # reward weighting
        self._orientation_weight = 10.0
        self._ctrl_cost_weight = 0 #1e-4
        self._position_weight[leg_indices] = 1 #5.0
        self._position_weight[waist_index] = 1e-1 #3.0
        self._position_weight[arm_indices] = 1e-2 #1.0
        self._joint_vel_weight = 1e-4
        self._angular_vel_weight = 1e-4
        self._action_smoothness_weight = 0 #1e-4

        self._include_previous_actions = include_previous_actions
        self._action_smoothness = action_smoothness
        if self._action_smoothness and not self._include_previous_actions:
            raise ValueError("`action_smoothness` penalty can only be used if previous actions are included in the observation space.")
        self._previous_action = None
        
        # Get the path to the XML file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        xml_dir = os.path.join(cur_dir, "unitree_robots", "g1")
        xml_path = os.path.join(xml_dir, "scene_23dof.xml")

        MujocoEnv.__init__(
            self,
            xml_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=800,
            height=600,
            render_mode=render_mode,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        # Get default joint positions from the stand keyframe
        self._get_default_joint_pos_from_stand()
        
        # Calculate observation indices for reward function
        self._obs_indices = self._calculate_obs_indices()
        
        # Calculate observation space size
        # base_acceleration + base_quat (4) + base_orientation (3) + projected_gravity (3) + angular_velocity (3) + joint_pos (23) + joint_vel (23) + optional previous_actions (23)

        acc_size = 3
        quat_size = 4
        orientation_size = 3
        gravity_orientation_size = 3
        angular_vel_size = 3
        joint_pos_size = 23
        joint_vel_size = 23
        previous_actions_size = 23 if self._include_previous_actions else 0
        
        obs_size = acc_size + quat_size + orientation_size + gravity_orientation_size + angular_vel_size + joint_pos_size + joint_vel_size + previous_actions_size

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        
        self._normalized_actions = normalized_actions
        self._original_action_space = self.action_space
        if self._normalized_actions:
            # Create normalized action space [-1, 1]
            self.action_space = Box(
                low=-1.0, high=1.0, shape=(self._original_action_space.shape[0],), dtype=np.float32
            )

        # used for LQG state estimate
        self.external_controller = None


    @staticmethod
    def add_args(parser):
        parser.add_argument('--no_normalized_actions', action='store_false', dest='normalized_actions', default=True)
        parser.add_argument('--include_previous_actions', action='store_true', dest='include_previous_actions', default=True)

    
    @staticmethod
    def get_env_kwargs(args):
        kwargs = {'normalized_actions': args.normalized_actions,
                  'include_previous_actions': args.include_previous_actions,
                  }

        return kwargs

    def is_healthy(self, obs):
        r, p, _ = obs[self._obs_indices['orientation_start']:self._obs_indices['orientation_end']]
        within_pitch = abs(p) < self._healthy_rp_range
        within_roll = abs(r) < self._healthy_rp_range
        is_healthy = within_roll and within_pitch
        return is_healthy

    def _calculate_obs_indices(self):
        """Calculate the indices for different observation components."""
        indices = {}


        indices['base_acc_start'] = 0
        indices['base_acc_end'] = 3

        indices['quat_start'] = indices['base_acc_end']
        indices['quat_end'] = indices['base_acc_end'] + 4

        indices['orientation_start'] = indices['quat_end']
        indices['orientation_end'] = indices['quat_end'] + 3

        # Angular velocity
        indices['angular_vel_start'] = indices['orientation_end']
        indices['angular_vel_end'] = indices['orientation_end'] + 3
        
        # Projected gravity
        indices['gravity_start'] = indices['angular_vel_end']
        indices['gravity_end'] = indices['angular_vel_end'] + 3

        # Joint positions
        indices['joint_pos_start'] = indices['gravity_end']
        indices['joint_pos_end'] = indices['gravity_end'] + 23
        
        # Joint velocities
        indices['joint_vel_start'] = indices['joint_pos_end']
        indices['joint_vel_end'] = indices['joint_pos_end'] + 23
        
        # Previous actions (optional)
        if self._include_previous_actions:
            indices['actions_start'] = indices['joint_vel_end']
            indices['actions_end'] = indices['joint_vel_end'] + 23
        else:
            indices['actions_start'] = None
            indices['actions_end'] = None
        
        return indices

    def _get_default_joint_pos_from_stand(self):
        """Get default joint positions from the stand keyframe."""
        # reset to stand keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # skip first 7 elements which are base position/orientation)
        self._default_joint_pos = self.data.qpos[7:].copy()

    def _get_obs(self):

        # measurements from IMU data (base/pelvis frame)
        base_quat = self.data.qpos[3:7] 
        angular_velocity = self.data.qvel[3:6]

        base_orientation = quat_to_euler(base_quat)

        # Projected gravity (3 elements)
        gravity_orientation = get_gravity_orientation(base_quat)
        
        quat_xyzw = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        
        base_acc_imu = self.data.qacc[0:3]
        R = Rotation.from_quat(quat_xyzw)
        # acceleration in world frame
        base_acc = R.apply(base_acc_imu)

        # Joint states
        joint_pos = self.data.qpos[7:] 
        joint_vel = self.data.qvel[6:] 
        
        # Relative joint positions
        relative_joint_pos = joint_pos - self._default_joint_pos
        
        if self._include_previous_actions:
            if self._previous_action is not None:
                previous_actions = self._previous_action
            else:
                previous_actions = np.zeros(23)  # 23 joints
        else:
            previous_actions = np.array([])
        
        # Concatenate all components in the specified order
        obs_components = [base_acc, base_quat, base_orientation, angular_velocity, gravity_orientation, relative_joint_pos, joint_vel]
        if self._include_previous_actions:
            obs_components.append(previous_actions)
        
        return np.concatenate(obs_components)

    def _get_reward(self, observation, action, terminated=False):
        # keep torques small if possible
        quad_ctrl_cost = -self._ctrl_cost_weight * np.square(action).sum()
        
        # smoothness penalty on actions (optional flag)
        if self._action_smoothness and self._previous_action is not None:
            action_smoothness_cost = -self._action_smoothness_weight * np.square(action - self._previous_action).sum()
        else:
            action_smoothness_cost = 0.0
        
        # projected gravity
        # gravity_orientation = observation[self._obs_indices['gravity_start']:self._obs_indices['gravity_end']]
        # unit_gravity_desired = np.array([0., 0., -1.])
        # gravity_error = gravity_orientation - unit_gravity_desired

        orientation = observation[self._obs_indices['orientation_start']:self._obs_indices['orientation_end']]

        # penalize gravity orientation error using MSE
        orientation_cost = -self._orientation_weight * np.sum(orientation[:2]**2)

        # Angular velocity penalty
        angular_vel = observation[self._obs_indices['angular_vel_start']:self._obs_indices['angular_vel_end']]
        angular_vel_cost = -self._angular_vel_weight * np.square(angular_vel).sum()

        # Joint position cost using cosine similarity
        relative_joint_positions = observation[self._obs_indices['joint_pos_start']:self._obs_indices['joint_pos_end']]
        wrapped_error = np.atan2(np.sin(relative_joint_positions), np.cos(relative_joint_positions)) 
        joint_pos_cost = -np.sum(self._position_weight * np.square(wrapped_error))
        
        # Compute cosine similarities (1 = perfect alignment, -1 = opposite)
        # joint_cosine_similarities = cosine_similarity_angles(relative_joint_positions)
        # joint_pos_cost = -np.sum(self._position_weight * (1 - joint_cosine_similarities))


        # joint velocity cost (target velocity is 0)
        joint_velocities = observation[self._obs_indices['joint_vel_start']:self._obs_indices['joint_vel_end']]
        joint_vel_cost = -self._joint_vel_weight * np.square(joint_velocities).sum()  

        survival_reward = 0
        if not terminated:
            survival_reward = 2.
        
        # print("#####")
        # print("Orientation Cost:", orientation_cost)
        # print("Position Cost:", joint_pos_cost)
        # print("Angular Velocity Cost:", angular_vel_cost)
        # print("Joint Position Cost:", joint_pos_cost)
        # print("Joint Velocity Cost:", joint_vel_cost)
        # print("Control Cost:", quad_ctrl_cost)
        # print("Action Smoothness Cost:", action_smoothness_cost)
        return survival_reward + orientation_cost + angular_vel_cost + joint_pos_cost + joint_vel_cost + quad_ctrl_cost + action_smoothness_cost

    def set_controller(self, controller):
        '''
        Sets an external controller. Useful if the controller contains an observer that has a state estimate.
        Assumes the controller has a reset() function, otherwise an error will be thrown.
        The reset function must take in a parameter for the observation/noisy measurements to initialize the state estimate.
        '''
        self.external_controller = controller

    def step(self, action):
        if self._normalized_actions:
            # actions are in [-1, 1], so we must unnormalize
            action_scale = (self._original_action_space.high + self._original_action_space.low) / 2
            action_range = (self._original_action_space.high - self._original_action_space.low) / 2
            scaled_action = action * action_range + action_scale
        else:
            scaled_action = action
        

        clipped_action = np.clip(scaled_action, self._original_action_space.low, self._original_action_space.high)

        # step in mujoco
        self.do_simulation(clipped_action, self.frame_skip)
        # get updated obs
        observation = self._get_obs()
        terminated = not self.is_healthy(observation)
        reward = self._get_reward(observation, clipped_action, terminated)
        info = {}
        
        if self._include_previous_actions:
            self._previous_action = clipped_action.copy()

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        # use the "stand" keyframe for initial state (id=0)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        xy_noise_scale = 0.01
        z_noise_scale = 0.001
        quat_noise_scale = 0.01
        joint_pos_noise_scale = 0.01

        xy_noise = self.np_random.uniform(
            low=-xy_noise_scale, 
            high=xy_noise_scale, 
            size=2
        )
        z_noise = self.np_random.uniform(
            low=-z_noise_scale, 
            high=z_noise_scale, 
            size=1
        )
        quat_noise = self.np_random.uniform(
            low=-quat_noise_scale, 
            high=quat_noise_scale, 
            size=4
        )
        base_pos_noise = np.concatenate([xy_noise, z_noise, quat_noise])
        
        joint_pos_noise = self.np_random.uniform(
            low=-joint_pos_noise_scale, 
            high=joint_pos_noise_scale, 
            size=self.model.nq - 7
        )
        
        qpos_noise = np.concatenate([base_pos_noise, joint_pos_noise])
        qpos = self.data.qpos + qpos_noise
        
        qvel = self.data.qvel
        self.set_state(qpos, qvel)


        self._previous_action = None

        observation = self._get_obs()

        # optionally reset the controller (useful when observer has state estimate)
        if self.external_controller is not None:
            # TODO add gaussian noise to sensors in observation? Match noise from lqg
            base_quat_meas = observation[self._obs_indices['quat_start']:self._obs_indices['quat_end']]
            base_angular_velocity_meas = observation[self._obs_indices['angular_vel_start']:self._obs_indices['angular_vel_end']]
            joint_pos_meas = observation[self._obs_indices['joint_pos_start']:self._obs_indices['joint_pos_end']] + self._default_joint_pos
            joint_vel_meas = observation[self._obs_indices['joint_vel_start']:self._obs_indices['joint_vel_end']]
            measurements = np.concatenate([joint_pos_meas, joint_vel_meas, base_quat_meas, base_angular_velocity_meas])
            self.external_controller.reset(measurements)

        return observation


if __name__ == '__main__':

    env = G1BalanceEnv(render_mode="human")
    obs, info = env.reset()

    terminated = False
    total_reward = 0.0
    while not terminated:
        a = env.action_space.sample()
        a = np.zeros_like(a)
        obs, reward, terminated, truncate, info = env.step(a)
        total_reward += reward
    print("Episode return:", total_reward)
    env.close()