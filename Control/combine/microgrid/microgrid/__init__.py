from gym.envs.registration import register

register(id='MicrogridControlGym-v0',
    entry_point='microgrid.envs:microgrid_control_gym',
)

register(id='MicrogridControlGym-v1',
    entry_point='microgrid.envs:microgrid_control_gym_test',
)

register(id='MicrogridControlGymExcess-v0',
    entry_point='microgrid.envs:microgrid_control_gym2',
)
