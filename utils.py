import sys, os
import torch
import numpy as np
import random
from stable_baselines3.common import vec_env

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # GPU seeding
    torch.cuda.manual_seed(seed)
    # Deterministic operations for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cur_path():
    '''
    Return the current directory of frontier policy optimization.
    '''
    return os.path.dirname(os.path.realpath(__file__))