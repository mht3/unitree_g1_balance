from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import vec_env
import numpy as np
import os

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Taken from SB3 help docs: 
    
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, name: str = 'best_model'):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, name)
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            if not os.path.exists(self.log_dir):
                return True
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)
        return True

def compare_models(model_1, model_2):
    '''
    Compare two torch models. Raises a value error if the model architectures differ in any way.
    model_1: RL policy network
    model_2: pretrained policy network
    '''
    state_dict_1 = model_1.state_dict()
    state_dict_2 = model_2.state_dict()

    keys_1 = set(state_dict_1.keys())
    keys_2 = set(state_dict_2.keys())

    if keys_1 != keys_2:
        missing_1 = keys_1 - keys_2
        missing_2 = keys_2 - keys_1
        raise ValueError(
            f"Model layer mismatch.\n"
            f"In model_1 but not in model_2: {missing_1}\n"
            f"model_1 is missing: {missing_2}"
        )

    for key in keys_1:
        param_1 = state_dict_1[key]
        param_2 = state_dict_2[key]

        if param_1.shape != param_2.shape:
            raise ValueError(
                f"Shape mismatch at layer '{key}': "
                f"{param_1.shape} (model_1) vs {param_2.shape} (model_2)"
            )

def model_inference(obs, model):
    action, _ = model.predict(obs)
    return action

def run_single_episode(model, env, parameter=None):
    terminate = False
    truncate = False
    # Reset Environment
    if isinstance(env, vec_env.VecEnv):
        obs = env.reset()
    else:
        obs, info = env.reset()
    rew = 0
    while not truncate and not terminate:
        if parameter is None:
            action = model(obs)
        else:
            action = model(obs, parameter)
        if isinstance(env, vec_env.VecEnv):
            obs, reward, terminate, info = env.step(action)
        else:
            obs, reward, terminate, truncate, info = env.step(action)
        rew += reward
    return rew