#from gym.envs.registration import register
#from .Reinforcement_learning.envs.microgrid_control_gymenv2 import microgrid_control_gym
#from .Reinforcement_learning.envs.microgrid_test_control_gymenv import microgrid_control_gym_test



#register(id='MicrogridEnv-v0',entry_point="microgridenv:microgrid_control_gymenv2:microgrid_control_gym")

from setuptools import setup 
setup(name='Reinforcement_learning', version='0.0.1', install_requires=['gym'] )
