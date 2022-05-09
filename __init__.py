from gym.envs.registration import register

register(id='MicrogridControlGym-v0',
    entry_point='Reinforcement_learning.envs:MicrogridControlGym',
)

from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

register(id='MicrogridEnv-v0',entry_point="gym.envs.microgridenv:microgrid_control_gym")## Fait comme le fichier dans gym
register(id='MicrogridEnv-v1',entry_point="gym.envs.microgridenv:microgrid_control_gym_test") ##Fait comme le fichier dans gym
