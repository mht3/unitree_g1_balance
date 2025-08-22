import sys, os, shutil
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
import torch
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from trainer import Trainer
from algorithm_utils import SaveOnBestTrainingRewardCallback, compare_models


class SACTrainer(Trainer):

    @staticmethod
    def add_args(parser):
        parser.add_argument("-w", "--warm_start_path", help='Path for warm start policy.zip file. Must be same architecture as RL policy network.', type=str, default=None)
        parser.add_argument("-t", "--timesteps", help='number of timesteps for training', default=1e6, type=int)
        parser.add_argument("--lr", help='learning rate', default=3e-4, type=float)
        parser.add_argument('--policy_net',
                            nargs='*',
                            type=int,
                            default=[256, 256],
                            help='A list of integers representing the sizes of hidden layers.,e.g., --policy_net 32 64 128. Default: [64, 64]')
        parser.add_argument('--q_net',
                        nargs='*',
                        type=int,
                        default=[256, 256],
                        help='A list of integers representing the sizes of hidden layers of the Q-network.')
        parser.add_argument('--buffer_size', type=int, default=1000000, help='SAC Replay Buffer Size')
        parser.add_argument('--learning_starts', type=int, default=100, help='Number of learning starts,')
        parser.add_argument('--batch_size', type=int, default=256, help='Minibatch size. Default 256')
        parser.add_argument('--tau', type=float, default=0.005, help='the soft update coefficient') 
        parser.add_argument('--stats_window_size', type=int, default=100, help='Number of episodes for rollout logging. E.g. average episode reward plot over 100 episodes.')

    @staticmethod
    def get_training_kwargs(args):
        model_kwargs = {'buffer_size': args.buffer_size,
                        'learning_starts': args.learning_starts,
                        'batch_size': args.batch_size,
                        'tau': args.tau,
                        'stats_window_size': args.stats_window_size,
                        'seed': args.seed # already defined in main.py
                        }

        policy_net_str = "-".join(map(str, args.policy_net))
        q_net_str = "-".join(map(str, args.q_net))

        log_name = "{}_pi_{}_qf_{}_s_{}".format(args.algorithm, policy_net_str, q_net_str, args.seed)
        
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
                    'q_net_list': args.q_net,
                    'warm_start_path': args.warm_start_path
                }
        return training_kwargs

    def train(self, env, model_path, model_kwargs=None, policy_net_list=None, q_net_list=None, 
              timesteps=1e6, log_name='sac', warm_start_path=None):
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                             net_arch=dict(pi=policy_net_list, qf=q_net_list))

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
            raise NotImplementedError("Not yet implemented.")

        # check model kwargs is not none
        if model_kwargs is None:
            model_kwargs = {}
        model = SAC("MlpPolicy", env=env, **model_kwargs, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=tensorboard_log)

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
        return SAC.load(model_path)