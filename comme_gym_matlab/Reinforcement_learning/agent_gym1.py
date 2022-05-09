import argparse
parser = argparse.ArgumentParser(description='Agent qui interagit en DQN')
parser.add_argument("--hydrogene_storage_penality", help="activate rewards (or penalities) linked to hydrogene storage", const=True, nargs='?', default=False)
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



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve',fname=None):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=2)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed"+fname)
    plt.savefig("Result/" + fname)
    # plt.show()



if __name__=="__main__":


    hydrogene_storage_penality = ""
    ppc = ""
    sell_grid = ""
    if args.hydrogene_storage_penality:
        hydrogene_storage_penality = " Hydrogene Storage Penality "
    if args.ppc is not None:
        ppc = " PPC max cons=" + str(ppc)
    if args.sell_to_grid:
        sell_grid = " Excess energy goes to main grid "


    fname=hydrogene_storage_penality+ppc+sell_grid
    # Create unique log dir
    log_dir = "tmp/gym/{}".format(int(time.time()))

    os.makedirs(log_dir, exist_ok=True)

    len_episode=8760
    num_episode=35
    env=microgrid_control_gym(data="train", hydrogene_storage_penalty=args.hydrogene_storage_penality, ppc=args.ppc, sell_to_grid=args.sell_to_grid, total_timesteps=len_episode*num_episode)

    env = Monitor(env, log_dir, allow_early_resets=False)
    env = DummyVecEnv([lambda: env])
    callback = SaveOnBestTrainingRewardCallback(check_freq=len_episode, log_dir=log_dir)
    model=DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=len_episode*num_episode, callback=callback)
    # model.learn(total_timesteps=1752000)
    # done=False
    # for i in range (20):
    #     obs=env.reset()
    #     while done==False:
    #         action, state = model.predict(obs)
    #         obs, reward,done, info = env.step(action)
    from stable_baselines import results_plotter

    # Helper from the library
    results_plotter.plot_results([log_dir], len_episode*num_episode, results_plotter.X_TIMESTEPS, "DQN Microgrid Control")
    # plt.show()
    model.save("deepq_microgrid_control")
    plot_results(log_dir,fname=fname)


