import argparse
parser = argparse.ArgumentParser(description='Agent qui interagit en DQN')
parser.add_argument("--hydrogene_storage_penality", help="activate rewards (or penalities) linked to hydrogene storage", const=True, nargs='?', default=False)
parser.add_argument("--agent_to_test", help="specify the trained agent name", default="deepq_microgrid_control")#, const= 'deepq_microgrid_control')
parser.add_argument("--sell_to_grid", help="activate selling transition with main grid", const=True, nargs='?', default=False)
parser.add_argument("--ppc", help="PPC maximum constraint, can be a number or None if you don't want limits", const=1.5, nargs='?', default=None)
parser.add_argument("--env", help = "The environment to import, can be by default microgrid_control_gymenv2 or if called microgrid_control_gymenv2_excess", const='excess',nargs='?',default=None)
args = parser.parse_args()
print(args)

import gym
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import time
from stable_baselines.common.atari_wrappers import make_atari
if args.env!=None:
    from envs.microgrid_control_gymenv2_excess import microgrid_control_gym
else:
    from envs.microgrid_control_gymenv2 import microgrid_control_gym
from stable_baselines import DQN
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from Callbacks.SavingBestRewards import SaveOnBestTrainingRewardCallback

def predict():
    model=DQN.load(args.agent_to_test)
    test_env = microgrid_control_gym(data="test", hydrogene_storage_penalty=args.hydrogene_storage_penality, ppc=args.ppc,
                                sell_to_grid=args.sell_to_grid, total_timesteps=len_episode)
    test_env = Monitor(test_env, log_dir, allow_early_resets=True)
    test_env = DummyVecEnv([lambda: test_env])
    obs=test_env.reset()
    while True:
        action,_states = model.predict(obs)
        obs,reward,dones,info=test_env.step(action)


if __name__=="__main__":
    predict()