from gymnasium.envs.registration import register

register(
    id='G1Balance-v0',
    entry_point='environments.g1_balance:G1BalanceEnv',
    max_episode_steps=500, # max time = 10 seconds. dt_ctrl = 0.02
)

# Dictionary of all environment classes that require custom command line arguments.
# See G1Balance-v0 `add_args` and `get_env_kwargs` methods.
CUSTOM_ENV_CLASSES = {
    'G1Balance-v0': 'environments.g1_balance:G1BalanceEnv',
}