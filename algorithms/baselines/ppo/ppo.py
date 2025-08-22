import sys, os, shutil
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
import torch
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from trainer import Trainer
from algorithm_utils import SaveOnBestTrainingRewardCallback, compare_models


class PPOTrainer(Trainer):

    @staticmethod
    def add_args(parser):
        parser.add_argument("-w", "--warm_start_path", help='Path for warm start policy.zip file. Must be same architecture as RL policy network.', type=str, default=None)
        parser.add_argument("-t", "--timesteps", help='number of timesteps for training', default=1e6, type=int)
        parser.add_argument("--lr", help='learning rate', default=3e-4, type=float)
        parser.add_argument('--policy_net',
                            nargs='*',
                            type=int,
                            default=[64, 64],
                            help='A list of integers representing the sizes of hidden layers.,e.g., --policy_net 32 64 128. Default: [64, 64]')
        parser.add_argument('--value_net',
                        nargs='*',
                        type=int,
                        default=[64, 64],
                        help='A list of integers representing the sizes of hidden layers of the value network.')
        parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs for PPO surrogate loss (after each rollout is collected). Default: 10')
        parser.add_argument('--n_steps', type=int, default=2048, help='The number of steps to run for each environment per update. Default: 2048')
        parser.add_argument('--batch_size', type=int, default=64, help='Minibatch size. Default 64')
        parser.add_argument('--gae_lambda', type=float, default=0.95, help='TD(lambda) value for GAE.') 
        parser.add_argument('--ent_coef', type=float, default=0.0, help='Entropy weighted loss term.') 
        parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function weighted loss term.') 
        parser.add_argument('--stats_window_size', type=int, default=100, help='Number of episodes for rollout logging. E.g. average episode reward plot over 100 episodes.')
        
    @staticmethod
    def get_training_kwargs(args):
        model_kwargs = {'n_epochs': args.n_epochs,
                        'n_steps': args.n_steps,
                        'batch_size': args.batch_size, 
                        'learning_rate': args.lr,
                        'gae_lambda': args.gae_lambda,
                        'ent_coef': args.ent_coef,
                        'vf_coef': args.vf_coef,
                        'stats_window_size': args.stats_window_size,
                        'seed': args.seed # already defined in main.py
                        }

        policy_net_str = "-".join(map(str, args.policy_net))
        value_net_str = "-".join(map(str, args.value_net))

 
        log_name = "{}_pi_{}_vf_{}_s_{}".format(args.algorithm, policy_net_str, value_net_str, args.seed)
        
        if args.warm_start_path is not None:
            args.warm_start_path = os.path.abspath(args.warm_start_path)
            if not os.path.exists(args.warm_start_path):
                raise ValueError("Path `{}` does not exist for warm start model.".format(args.warm_start_path))
            log_name = log_name + '_warm_start'
        training_kwargs = {
                    'timesteps': args.timesteps,
                    'model_kwargs': model_kwargs,
                    'log_name': log_name,
                    'policy_net_list': args.policy_net,
                    'value_net_list': args.value_net,
                    'warm_start_path': args.warm_start_path,
                }
        return training_kwargs

    def train(self, env, model_path, model_kwargs=None, policy_net_list=None, value_net_list=None, 
              timesteps=1e6, log_name='ppo', warm_start_path=None):


        # Use standard MLP policy
        policy_class = "MlpPolicy"
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                                net_arch=dict(pi=policy_net_list, vf=value_net_list))
        if self.logger is not None:
            log_model_path = os.path.join(self.cur_path, "logs", log_name)
            tensorboard_log = os.path.join(self.cur_path, "tb/")
            os.makedirs(log_model_path, exist_ok=True)
            env = Monitor(env, log_model_path)
            # setup callbacks
            wandb_callback = WandbCallback()
            model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_model_path, name=self.logger.id)
            callbacks = CallbackList([wandb_callback, model_save_callback])
        else:
            tensorboard_log = None
            callbacks = None
        if warm_start_path is not None:
            print("Loading pretrained policy network for warm start...")
            pretrained_model = ActorCriticPolicy.load(warm_start_path)

        # check model kwargs is not none
        if model_kwargs is None:
            model_kwargs = {}
        model = PPO(policy_class, env=env, **model_kwargs, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log)

        if warm_start_path is not None:
            # compare models and ensure they are identical, otherwise raise error
            compare_models(model.policy, pretrained_model)
            model.policy.load_state_dict(pretrained_model.state_dict())
            with torch.no_grad():
                if hasattr(model.policy, 'log_std'):
                    # Can explore different initializations on standard deviation of policy
                    # model.policy.log_std.data.fill_(-1)  # std = exp(-1) = 0.368
                    print("Warning: Starting with policy log std: ", model.policy.log_std.data)

        # begin training
        model.learn(total_timesteps=timesteps, callback=callbacks, tb_log_name=log_name)
        if self.logger is not None:
            # Check if the callback saved a best model
            expected_model_path = os.path.join(log_model_path, self.logger.id + ".zip")
            if not os.path.exists(expected_model_path):
                raise ValueError('No best model was saved by the callback.')
            
            # copy model into models folder
            shutil.copyfile(expected_model_path, model_path)
        return model

    @staticmethod
    def load(model_path):
        return PPO.load(model_path)