from gymnasium.envs.registration import register

register(
    id='G1Balance-v0',
    entry_point='environments.g1.g1_balance:G1BalanceEnv',
    max_episode_steps=200,
)