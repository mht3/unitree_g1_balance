import numpy as np
import os

from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8)),
    "elevation": -20.0,
}


def mass_center(model, data):
    """Calculate the center of mass of the robot."""
    mass = np.expand_dims(model.body_mass, axis=1)
    xpos = data.xipos
    return (np.sum(mass * xpos, axis=0) / np.sum(mass))[0:2].copy()


class G1BalanceEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description

    This environment is for the Unitree G1 humanoid robot to learn balancing.
    The robot has 23 degrees of freedom and uses a floating base (pelvis) as the root joint.
    The goal is to maintain balance while standing.

    ### Action Space
    The action space is a `Box(-1, 1, (23,), float32)`. An action represents the torques applied at the hinge joints.

    ### Observation Space
    Observations consist of positional values of different body parts of the G1 robot,
    followed by the velocities of those individual parts (their derivatives) with all the
    positions ordered before all the velocities.

    By default, observations do not include the x- and y-coordinates of the pelvis. These may
    be included by passing `exclude_current_positions_from_observation=False` during construction.

    ### Rewards
    The reward consists of:
    - *healthy_reward*: Every timestep that the robot is alive (see section Episode Termination for definition), it gets a reward of fixed value `healthy_reward`
    - *balance_reward*: A reward for maintaining balance (minimal movement of center of mass)
    - *ctrl_cost*: A negative reward for penalising the robot if it has too large of a control force.

    ### Starting State
    The robot starts in the "stand" keyframe position with some noise added for stochasticity.

    ### Episode End
    The robot is said to be unhealthy if the z-position of the pelvis is no longer contained in the
    closed interval specified by the argument `healthy_z_range`.

    ### Arguments
    | Parameter                                    | Type      | Default          | Description                                                                                                                                                               |
    | -------------------------------------------- | --------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"scene_23dof.xml"` | Path to a MuJoCo model                                                                                                                                                    |
    | `balance_reward_weight`                      | **float** | `1.0`            | Weight for _balance_reward_ term (see section on reward)                                                                                                                  |
    | `ctrl_cost_weight`                           | **float** | `0.1`            | Weight for _ctrl_cost_ term (see section on reward)                                                                                                                       |
    | `healthy_reward`                             | **float** | `5.0`            | Constant reward given if the robot is "healthy" after timestep                                                                                                         |
    | `terminate_when_unhealthy`                   | **bool**  | `True`           | If true, issue a done signal if the z-coordinate of the pelvis is no longer in the `healthy_z_range`                                                                       |
    | `healthy_z_range`                            | **tuple** | `(0.6, 1.2)`     | The robot is considered healthy if the z-coordinate of the pelvis is in this range                                                                                      |
    | `reset_noise_scale`                          | **float** | `1e-2`           | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                            |
    | `exclude_current_positions_from_observation` | **bool**  | `True`           | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 60,
    }

    def __init__(
        self,
        xml_file="scene_23dof.xml",
        balance_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.6, 1.2),
        reset_noise_scale=1e-2,
        exclude_current_positions_from_observation=True,
        **kwargs
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            balance_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs
        )

        self._balance_reward_weight = balance_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        # Calculate observation space size
        # G1 has 30 qpos (7 for floating base + 23 joints) and 29 qvel (6 for floating base + 23 joints)
        # If excluding positions, we remove the first 2 elements (x, y) from qpos
        if exclude_current_positions_from_observation:
            obs_size = (30 - 2) + 29  # 28 + 29 = 57
        else:
            obs_size = 30 + 29  # 59

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # Get the path to the XML file
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        xml_dir = os.path.join(cur_dir, "unitree_robots", "g1")
        xml_path = os.path.join(xml_dir, xml_file)

        MujocoEnv.__init__(
            self, xml_path, 5, observation_space=observation_space, **kwargs
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            # Remove x, y coordinates from position (keep z and orientation)
            position = position[2:]

        return np.concatenate((position, velocity))

    def step(self, action):
        xy_position_before = mass_center(self.model, self.data)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = mass_center(self.model, self.data)

        # Calculate balance reward (minimize movement)
        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        balance_reward = -self._balance_reward_weight * np.linalg.norm(xy_velocity)

        ctrl_cost = self.control_cost(action)
        healthy_reward = self.healthy_reward

        rewards = balance_reward + healthy_reward

        observation = self._get_obs()
        reward = rewards - ctrl_cost
        terminated = self.terminated
        info = {
            "reward_balance": balance_reward,
            "reward_quadctrl": -ctrl_cost,
            "reward_alive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": xy_velocity[0],
            "y_velocity": xy_velocity[1],
            "balance_reward": balance_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        # Use the "stand" keyframe (id=0) as mentioned in the user query
        self.data.qpos[:] = self.model.key_qpos[0].copy()
        self.data.qvel[:] = self.model.key_qvel[0].copy()

        # Add noise for stochasticity
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.data.qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.data.qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


if __name__ == '__main__':
    import time

    env = G1BalanceEnv(render_mode="human")
    obs, _ = env.reset()

    done = False
    ep_ret = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, _ = env.step(a)
        ep_ret += r
        # time.sleep(0.04)
    print("Episode return:", ep_ret)