import os
import sys
import numpy as np
# import gym
import gymnasium as gym
import mujoco
import mujoco.viewer

class G1BalanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 200}

    def __init__(self, render_mode=None):

        cur_dir = os.path.dirname(os.path.abspath(__file__))
        xml_dir = os.path.join(cur_dir, "..", "unitree_robots", "g1")
        xml_path = os.path.join(xml_dir, "scene_23dof.xml")

        # Load model & data
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Viewer
        self.viewer = None
        self.render_mode = render_mode

        # Action space: torques for 12 lower-body joints
        torque_limit = 30.0  # Nm (adjust if needed)
        self.action_space = gym.spaces.Box(
            low=-torque_limit, high=torque_limit, shape=(12,), dtype=np.float32
        )

        # Observation space: orientation, angular vel, joint pos/vel
        obs_dim = 6 + 3 + 12 + 12  # [quat, ang_vel, qpos, qvel]
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, shape=(obs_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        # Small random perturbation in qpos/qvel
        self.data.qpos[:] += 0.01 * self.np_random.standard_normal(self.model.nq)
        self.data.qvel[:] += 0.01 * self.np_random.standard_normal(self.model.nv)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # Apply torques (first 12 joints)
        self.data.ctrl[:12] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Base orientation quaternion (w, x, y, z)
        quat = self.data.qpos[3:7].copy()
        # Angular velocity (world frame)
        ang_vel = self.data.qvel[3:6].copy()
        # Joint pos/vel (first 12 lower-body joints only)
        qpos_joints = self.data.qpos[7:19].copy()
        qvel_joints = self.data.qvel[6:18].copy()

        return np.concatenate([quat, ang_vel, qpos_joints, qvel_joints])

    def _compute_reward(self, obs):
        quat = obs[0:4]
        # Convert quaternion to pitch/roll approximation
        # Roll ~ 2*(w*x + y*z), Pitch ~ 2*(w*y - x*z)
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x+y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))

        upright_bonus = np.exp(-5*(roll**2 + pitch**2))
        alive_bonus = 1.0
        energy_penalty = -0.001 * np.sum(self.data.ctrl[:12]**2)

        return alive_bonus + upright_bonus + energy_penalty

    def _check_termination(self, obs):
        quat = obs[0:4]
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x+y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))

        # Update and read CoM
        mujoco.mj_comPos(self.model, self.data)
        com = self.data.subtree_com[0]  # root body CoM

        if com[2] < 0.1:
            return True
        return False

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


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