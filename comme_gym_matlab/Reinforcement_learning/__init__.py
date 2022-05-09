from gym.envs.registration import register

register(id='MicrogridControlGym-v0',
    entry_point='Reinforcement_learning.envs:MicrogridControlGym',
)
