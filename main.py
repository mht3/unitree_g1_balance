import os
import utils
import environments
import algorithms
import experts
import argparse
import importlib
import gymnasium as gym
import wandb
import numpy as np
import copy 

def parse_args():
    '''
    Parse command line arguments for training/testing a model.

    Returns:
        train (bool): Whether to train (True) or test (False) the model
        algorithm_class (Class): algorithm class located in the algorithms folder.
        model_path (str): Path where model will be saved/loaded
        training_kwargs (dict): Additional training arguments for the specified algorithm
        env_name (str): name of the environment to be loaded by gym. `gym.make(env_name)`
        env_kwargs (dict): Dictionary of environment parameters. 
        debug (bool): Flag for debugging training. Will turn wandb logging off.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, default='G1Balance-v0', help='Gym environment name.')
    parser.add_argument('--test', action='store_true', help='Flag for loading a model and testing on a specific environment. Otherwise training is set to True.')
    parser.add_argument("--algorithm", choices=['PPO', 'SAC'], help='Model to use. Each algorithm requires additional arguments. See `algorithms/` for more info.', required=True)
    parser.add_argument("--expert", choices=[None, 'G1LQG', 'G1LQR'], help='Expert to use. See `experts/` for more info.', default=None)
    parser.add_argument("--seed", help='random seed to use.', type=int, default=42)
    parser.add_argument('--render', action='store_true', help='Flag for rendering the environment.')
    parser.add_argument('--debug', action='store_true', help='Flag for debugging the training process. Turns wandb logging off.')
    parser.add_argument("--num_test_episodes", help='Number of test episodes to use for model testing.', type=int, default=5)
    parser.add_argument('--test_expert', action='store_true', help='Flag for testing an expert in a specified environment. Overrides all algorithm specific commands.')

    # Parse known arguments before parsing remainder of training args
    args, _ = parser.parse_known_args()

    # set seed
    utils.set_seed(args.seed)

    train = not args.test

    # add algorithm-specific arguments
    algorithm_name = args.algorithm + 'Trainer'
    algorithm_module = importlib.import_module('algorithms')
    algorithm_class = getattr(algorithm_module, algorithm_name)
    algorithm_class.add_args(parser)


    # environment specific arguments
    env_id = args.env_id
    env_kwargs = {}
    if args.render:
        env_kwargs['render_mode'] = 'human'

    if env_id in environments.CUSTOM_ENV_CLASSES:
        env_module_path, env_class_name = environments.CUSTOM_ENV_CLASSES[env_id].split(':')
        env_module = importlib.import_module(env_module_path)
        env_class = getattr(env_module, env_class_name)
        env_class.add_args(parser)

    # parse environment and algorithm specific arguments
    args = parser.parse_args()
    # expert
    if args.expert is not None:
        expert_module = importlib.import_module('experts')
        expert_class = getattr(expert_module, args.expert)
        # update expert to specific class for algorithm to optionally handle (see algorithms/BCTrainer for an example)
        args.expert = expert_class
    else:
        expert_class = None

    # parse training kwargs from algorithm
    algorithm_train_kwargs = algorithm_class.get_training_kwargs(args)
    # default log name on weights and biases
    default_log_name = args.algorithm + '_s_{}'.format(args.seed)
    log_name = algorithm_train_kwargs.get('log_name', default_log_name)

    # parse optional environment kwargs
    if env_id in environments.CUSTOM_ENV_CLASSES:
        env_specific_kwargs = env_class.get_env_kwargs(args)
        env_kwargs.update(env_specific_kwargs)

    # logging setup
    if args.debug or not train:
        logger = None
    else:
        config = copy.deepcopy(algorithm_train_kwargs)
        config['env_id'] = env_id
        config[env_id] = env_specific_kwargs
        logger = init_wandb(config, log_name)

    # Model Path for loading model
    cur_path = utils.get_cur_path()
    models_folder = os.path.join(cur_path, 'models', env_id)
    os.makedirs(models_folder, exist_ok=True)
    model_path = os.path.join(models_folder, log_name+'_best.zip')

    # parse algorithm init kwargs (algorithm dependent)
    algorithm_init_kwargs = algorithm_class.get_init_kwargs(args)
    # add to init kwargs (needed for base class)
    algorithm_init_kwargs['cur_path'] = cur_path
    algorithm_init_kwargs['logger'] = logger

    algorithm_params = (algorithm_class, algorithm_init_kwargs, algorithm_train_kwargs)
    expert_params = (expert_class, args.test_expert)
    return train, env_id, env_kwargs, algorithm_params, expert_params, model_path, args.num_test_episodes

def test_model(model, test_env, num_episodes=1):
    rewards = []
    for _ in range(num_episodes):
        reward = utils.run_single_episode(utils.model_inference, test_env, model)
        rewards.append(reward)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print("Reward: {:.2f} +/- {:.2f}".format(mean_reward, std_reward))
    test_env.close()
def init_wandb(config, log_name):
    # setup wandb
    run = wandb.init(
        project="g1_balance",
        name = log_name,
        config=config,
        sync_tensorboard=True,    
    )
    return run
    
def main():
    train, env_id, env_kwargs, algorithm_params, expert_params, model_path, num_test_episodes = parse_args()

    # environment initialization
    print("Loading environment...", end=' ')
    env = gym.make(env_id, **env_kwargs)
    print("Done.")

    # algorithm initialization
    algorithm_class, algorithm_init_kwargs, algorithm_train_kwargs = algorithm_params
    algorithm = algorithm_class(**algorithm_init_kwargs)

    # optional expert class logic
    expert_class, test_expert = expert_params
    if expert_class is not None and test_expert:
        # test expert and return
        print("Loading expert...", end=' ')
        expert = expert_class(env)
        print("Done.")
        # run test episodes and check reward
        print("Running {} test episode(s)...".format(num_test_episodes))
        test_model(expert, env, num_episodes=num_test_episodes)

        return

    if train:
        # train model
        print("Training model...")
        # all algorithms are expected to have 
        algorithm_train_args = (env, model_path)
        model = algorithm.train(*algorithm_train_args, **algorithm_train_kwargs)
    else:
        # Load model from path
        print("Loading model...", end=' ')
        model = algorithm.load(model_path)
        print("Done.")

    # run test episodes and check reward
    print("Running {} test episode(s)...".format(num_test_episodes))
    test_model(model, env, num_episodes=num_test_episodes)

if __name__ == '__main__':
    main()