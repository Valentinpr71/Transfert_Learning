from stable_baselines.deepq import DQN
from stable_baselines.bench import Monitor
from Callbacks.SavingBestRewards import SaveOnBestTrainingRewardCallback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
import gym
import os
import time
import argparse
parser = argparse.ArgumentParser(description='Agent qui interagit en DQN')
parser.add_argument("--hydrogene_storage_penality", help="activate rewards (or penalities) linked to hydrogene storage", const=True, nargs='?', default=False)
parser.add_argument("--sell_to_grid", help="activate selling transition with main grid", const=True, nargs='?', default=False)
parser.add_argument("--ppc", help="PPC maximum constraint, can be a number or None if you don't want limits", const=1.5, nargs='?', default=None)
parser.add_argument("--env", help = "The environment to import, can be by default microgrid_control_gymenv2 or if called microgrid_control_gymenv2_excess", const='excess',nargs='?',default=None)
args = parser.parse_args()
print(args)

class CustomDQNPolicy(MlpPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[256,256],)
                                           # layer_norm=False,
                                           # feature_extraction="mlp")

hydrogene_storage_penality = ""
ppc = ""
sell_grid = ""
if args.hydrogene_storage_penality:
    hydrogene_storage_penality = " Hydrogene Storage Penality "
if args.ppc is not None:
    ppc = " PPC max cons=" + str(ppc)
if args.sell_to_grid:
    sell_grid = " Excess energy goes to main grid "

fname = hydrogene_storage_penality + ppc + sell_grid
# Create unique log dir
log_dir = "tmp/gym/{}".format(int(time.time()))

os.makedirs(log_dir, exist_ok=True)

len_episode = 8760
num_episode = 35
env = gym.make("microgrid:MicrogridControlGym-v0")

# env=microgrid_control_gym(data="train", hydrogene_storage_penalty=args.hydrogene_storage_penality, ppc=args.ppc, sell_to_grid=args.sell_to_grid, total_timesteps=len_episode*num_episode)

env = Monitor(env, log_dir, allow_early_resets=False)
env = DummyVecEnv([lambda: env])
callback = SaveOnBestTrainingRewardCallback(check_freq=len_episode, log_dir=log_dir)
model=DQN(CustomDQNPolicy,env).load("deepq_microgrid_control",env=env)
model.learn(total_timesteps=len_episode*7, callback=callback)
