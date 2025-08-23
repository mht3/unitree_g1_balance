import numpy as np
import os

import mujoco
from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

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

def quat_rotate(quat, vec):
    """
    Rotate a vector by a quaternion.
    quat: [w, x, y, z]
    vec: [x, y, z]
    """
    w, x, y, z = quat
    vx, vy, vz = vec
    
    result = np.array([
        (1 - 2*y*y - 2*z*z) * vx + (2*x*y - 2*w*z) * vy + (2*x*z + 2*w*y) * vz,
        (2*x*y + 2*w*z) * vx + (1 - 2*x*x - 2*z*z) * vy + (2*y*z - 2*w*x) * vz,
        (2*x*z - 2*w*y) * vx + (2*y*z + 2*w*x) * vy + (1 - 2*x*x - 2*y*y) * vz
    ])
    
    return result

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

    - *base_pos (7 elements):* The position and orientation of the robot's base (pelvis), including x,y,z coordinates and quaternion orientation.
    - *base_vel (6 elements):* The linear and angular velocities of the base.
    - *joint_pos (23 elements):* The relative joint positions (deviation from default standing pose).
    - *joint_vel (23 elements):* The joint velocities.
    - *projected_gravity (3 elements, optional):* Gravity vector projected in the robot's base frame (included if `include_projected_gravity=True`).
    - *previous_actions (23 elements, optional):* Previous action values for action continuity (included if `include_previous_actions=True`).

    The observation space size depends on the flags:
    - Base: 59 elements (base_pos + base_vel + joint_pos + joint_vel)
    - With projected gravity: 62 elements (59 + 3)
    - With previous actions: 82 elements (59 + 23)
    - With both: 85 elements (59 + 3 + 23)

    | Num | Observation                                                                                                     | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)                |
    | --- | --------------------------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | -------------------------- |
    | 0   | x-coordinate of the pelvis                                                                                      | -Inf | Inf | floating_base_joint              | free  | position (m)               |
    | 1   | y-coordinate of the pelvis                                                                                      | -Inf | Inf | floating_base_joint              | free  | position (m)               |
    | 2   | z-coordinate of the pelvis                                                                                     | -Inf | Inf | floating_base_joint              | free  | position (m)               |
    | 3   | w-orientation of the pelvis                                                                                     | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 4   | x-orientation of the pelvis                                                                                     | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 5   | y-orientation of the pelvis                                                                                     | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 6   | z-orientation of the pelvis                                                                                     | -Inf | Inf | floating_base_joint              | free  | angle (rad)                |
    | 7   | x-coordinate velocity of the pelvis (center)                                                                     | -Inf | Inf | floating_base_joint              | free  | velocity (m/s)             |
    | 8   | y-coordinate velocity of the pelvis (center)                                                                     | -Inf | Inf | floating_base_joint              | free  | velocity (m/s)             |
    | 9   | z-coordinate velocity of the pelvis (center)                                                                     | -Inf | Inf | floating_base_joint              | free  | velocity (m/s)             |
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
    
    - *Survival reward:* +1.0 for staying alive
    - *Position penalty:* Penalty for deviating from reference pelvis position
    - *Orientation cost:* Penalty for non-upright orientation
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
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
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
        frame_skip=10,
        default_camera_config=DEFAULT_CAMERA_CONFIG,
        reset_noise_scale=1e-2,
        normalized_actions=True,
        include_projected_gravity=False,
        include_previous_actions=False,
        render_mode=None,
    ):
        utils.EzPickle.__init__(
            self,
            frame_skip,
            default_camera_config,
            reset_noise_scale,
            normalized_actions,
            include_previous_actions,
            include_projected_gravity,
            render_mode,
        )
        # z range of pelvis before termination (only care about lower bound)
        self._healthy_z_min = 0.2
        # healthy roll and pitch range
        # roll and pitch can't exceed 78.75 degrees
        self._healthy_rp_range = 7 * np.pi / 16
        self._reset_noise_scale = reset_noise_scale
        self._ctrl_cost_weight = 1e-4
        self._orientation_weight = 100.
        self._position_weight = 1.
        self._joint_vel_weight = 1e-4
        
        # Previous actions configuration
        self._include_previous_actions = include_previous_actions
        self._include_projected_gravity = include_projected_gravity
        self._previous_action = None
        self._action_rate_weight = 0.01  # Weight for action smoothness penalty
        
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
        # G1 has 30 qpos (7 for floating base + 23 joints) and 29 qvel (6 for floating base + 23 joints)

        base_pos_size = 7  # x,y,z + quaternion
        base_vel_size = 6  # linear + angular velocity
        joint_pos_size = 23  # joint positions
        joint_vel_size = 23  # joint velocities
        projected_gravity_size = 3 if self._include_projected_gravity else 0
        previous_actions_size = 23 if self._include_previous_actions else 0
        
        obs_size = base_pos_size + base_vel_size + joint_pos_size + joint_vel_size + projected_gravity_size + previous_actions_size

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


    @staticmethod
    def add_args(parser):
        parser.add_argument('--no_normalized_actions', action='store_false', dest='normalized_actions', default=True)
        parser.add_argument('--reset-noise-scale', type=float, default=1e-2)
        parser.add_argument('--include_previous_actions', action='store_true', dest='include_previous_actions', default=True)
        parser.add_argument('--include_projected_gravity', action='store_true', dest='include_projected_gravity', default=True)

    
    @staticmethod
    def get_env_kwargs(args):
        kwargs = {'reset_noise_scale': args.reset_noise_scale,
                  'normalized_actions': args.normalized_actions,
                  'include_previous_actions': args.include_previous_actions,
                  'include_projected_gravity': args.include_projected_gravity,
                  }

        return kwargs

    @property
    def is_healthy(self):
        above_min_height = (self._healthy_z_min < self.data.qpos[2])
        quat = self.data.qpos[3:7]
        r, p, _ = quat_to_euler(quat)
        within_pitch = abs(p) < self._healthy_rp_range
        within_roll = abs(r) < self._healthy_rp_range

        is_healthy = above_min_height and within_roll and within_pitch
        return is_healthy

    def _calculate_obs_indices(self):
        """Calculate the indices for different observation components."""
        indices = {}
        
        # Base position (always present) - 7 elements: x,y,z + quaternion
        indices['base_pos_start'] = 0
        indices['base_pos_end'] = 7
        
        # Base velocity (always present) - 6 elements: linear + angular velocity
        indices['base_vel_start'] = 7
        indices['base_vel_end'] = 13
        
        # Joint positions (always present) - 23 elements
        indices['joint_pos_start'] = 13
        indices['joint_pos_end'] = 36
        
        # Joint velocities (always present) - 23 elements
        indices['joint_vel_start'] = 36
        indices['joint_vel_end'] = 59
        
        # Projected gravity (optional)
        if self._include_projected_gravity:
            indices['gravity_start'] = 59
            indices['gravity_end'] = 62
            current_end = 62
        else:
            indices['gravity_start'] = None
            indices['gravity_end'] = None
            current_end = 59
        
        # Previous actions (optional)
        if self._include_previous_actions:
            indices['actions_start'] = current_end
            indices['actions_end'] = current_end + 23
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
        # Base state (position and velocity) - keep absolute
        base_pos = self.data.qpos[:7]  # 7 elements: x,y,z + quaternion
        base_vel = self.data.qvel[:6]  # 6 elements: linear + angular velocity
        
        # Joint states - make relative to default pose
        joint_pos = self.data.qpos[7:]  # 23 joint positions
        joint_vel = self.data.qvel[6:]  # 23 joint velocities
        
        # Relative joint positions (deviation from default pose)
        relative_joint_pos = joint_pos - self._default_joint_pos
        
        # Relative joint velocities
        relative_joint_vel = joint_vel
        
        # Combine base and joint states
        position = np.concatenate([base_pos, relative_joint_pos])
        velocity = np.concatenate([base_vel, relative_joint_vel])
        
        # Projected gravity (optional)
        if self._include_projected_gravity:
            quat = self.data.qpos[3:7]  # Pelvis orientation
            # Get gravity direction (normalized) - MuJoCo gravity is typically [0, 0, -9.81]
            gravity_magnitude = np.linalg.norm(self.model.opt.gravity)
            gravity_world = self.model.opt.gravity / gravity_magnitude  # Normalized gravity direction
            projected_gravity = quat_rotate(quat, gravity_world)
        else:
            projected_gravity = np.array([])
        
        # Include previous actions if flag is enabled
        if self._include_previous_actions:
            if self._previous_action is not None:
                previous_actions = self._previous_action
            else:
                previous_actions = np.zeros(23)  # 23 joints
        else:
            previous_actions = np.array([])
        
        # Concatenate all components
        obs_components = [position, velocity]
        if self._include_projected_gravity:
            obs_components.append(projected_gravity)
        if self._include_previous_actions:
            obs_components.append(previous_actions)
        
        return np.concatenate(obs_components)

    def _get_reward(self, observation, action):
        quad_ctrl_cost = -self._ctrl_cost_weight * np.square(action).sum()
        
        # smoothness penalty on actions
        if self._previous_action is not None:
            action_rate_cost = -self._action_rate_weight * np.square(action - self._previous_action).sum()
        else:
            action_rate_cost = 0.0
        
        # Orientation cost using Euler angles (roll, pitch, yaw) with less weight on yaw
        base_pos = observation[self._obs_indices['base_pos_start']:self._obs_indices['base_pos_end']]
        quat = base_pos[3:7]  # quaternion [w, x, y, z]
        roll, pitch, yaw = quat_to_euler(quat)
        
        # Penalize roll and pitch more heavily, yaw less
        orientation_cost = -self._orientation_weight * (roll**2 + pitch**2 + 0.01 * yaw**2)

        # joint velocity cost
        joint_velocities = observation[self._obs_indices['joint_vel_start']:self._obs_indices['joint_vel_end']]
        joint_vel_cost = -self._joint_vel_weight * np.square(joint_velocities).sum()  

        # pelvis position penalty (distance from origin)
        pelvis_pos = base_pos[0:3]
        pelvis_reference = np.array([0., 0., 0.79127])
        distance = np.sqrt(np.sum((pelvis_pos - pelvis_reference)**2))
        position_penalty = - self._position_weight * distance

        survival_reward = 1.0

        # print(position_penalty, orientation_cost, joint_vel_cost, quad_ctrl_cost)
        return survival_reward + position_penalty + orientation_cost + joint_vel_cost + quad_ctrl_cost + action_rate_cost

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
        reward = self._get_reward(observation, clipped_action)
        terminated = not self.is_healthy
        info = {}
        
        # Store current action as previous action for next step
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

        # Reset previous action
        self._previous_action = None

        observation = self._get_obs()
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