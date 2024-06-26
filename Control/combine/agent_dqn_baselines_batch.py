import argparse
import numpy as np
# parser = argparse.ArgumentParser(description='Agent qui interagit en DQN')
# parser.add_argument("--hydrogene_storage_penality", help="activate rewards (or penalities) linked to hydrogene storage", const=True, nargs='?', default=False)
# parser.add_argument("--sell_to_grid", help="activate selling transition with main grid", const=True, nargs='?', default=False)
# parser.add_argument("--ppc", help="PPC maximum constraint, can be a number or None if you don't want limits", const=1.5, nargs='?', default=None)
# #parser.add_argument("--env", help = "The environment to import, can be by default microgrid_control_gymenv2 or if called microgrid_control_gymenv2_excess", const='excess',nargs='?',default=None)
# parser.add_argument("--dim", help="Dimension of the PV arrays and Battery Capacity", const=np.array(12.,15.),nargs='?',default=np.array(12.,15.))
# args = parser.parse_args()
# print(args)
from microgrid.envs.import_data import Import_data

import gym
import matplotlib.pyplot as plt

import sys
import os
import time
import json
import hashlib

# if args.env!=None:
#     from envs.microgrid_control_gymenv2_excess import microgrid_control_gym
# else:
#     from envs.microgrid_control_gymenv2 import microgrid_control_gym
from stable_baselines.deepq import DQN
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from Callbacks.SavingBestRewards import SaveOnBestTrainingRewardCallback
from .Tuple_ech import Interact
from .buffer_tools.pretrain_policy import Pretrain
from .buffer_tools.hash_manager import Dim_manager
from .discrete_BCQ.main import main_BCQ
from stable_baselines.deepq.policies import FeedForwardPolicy


if __name__ == "__main__":
    import_data = Import_data(mode="train")
    consumption_norm = import_data._consumption_norm()
    consumption = import_data.consumption()

    distance_euclidienne = 4
    nb_voisins = 2
    dim_num_iteration = 50
    dim_boundaries = {'PV': {'low': 0,'high': 12},'batt': {'low': 0,'high': 15}}
    hydrogene_storage_penality = ""
    ppc = ""
    sell_grid = ""

    len_episode = 8760
    num_episode = 5

    manager = Dim_manager(distance=distance_euclidienne, nb_voisins=nb_voisins)
    manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)

    for i in range(dim_num_iteration):
        # On itère aléatoirement sur le dimensionnement du µgrid.
        dim=np.array([float(np.random.randint(dim_boundaries['PV']['low'],dim_boundaries['PV']['high'])),float(np.random.randint(dim_boundaries['batt']['low'],dim_boundaries['batt']['high']))])
        manager._dim(dim.tolist())
        # On s'intéresse aux parents du dimensionnement
        parents = manager.choose_parents()
        print("parents : ", parents, "dicto : ", manager.dicto,"dim : ", manager.dim)
        # Si l'on a pas déjà vu cet environnement, on l'ajoute au dictionnaire des dimensionnements connus et on poursuit la procédure.
        if manager.add_to_dicto():
            production_norm = import_data._production_norm(PV=dim[0])
            production = import_data.production()
            manager.add_data_prod(data_prod=production, data_prod_norm=production_norm)
            ####### Ici, le manager a toutes les informations possibles sur le dimensionnement. On peut couper le code pour matcher avec BCQ là


            log_dir, carry = manager.path()
            if carry:
                continue
            # S'il existe déjà un fichier "best_model" correspondant au dimensionnement en cours, alors ça ne sert à rien de poursuivre cette itération.
            BCQ = main_BCQ("microgrid:MicrogridControlGym-v0", manager=manager)






            #Maintenant deux possibilités: Soit l'agent a à disposition des autres agents déjà entraînés et il peut constituer un batch, soit non et on va l'entraîner en on-line off-policy.
            if len(parents)<2:#Pas assez de parents, on entraîne
                env = Monitor(env, log_dir, allow_early_resets=False)
                env = DummyVecEnv([lambda: env])
                callback = SaveOnBestTrainingRewardCallback(check_freq=len_episode, log_dir=log_dir)
                model = DQN(CustomDQNPolicy, env, verbose=1, exploration_final_eps=0.01, exploration_fraction=0.3,
                            target_network_update_freq=1095, buffer_size=1000000, batch_size=264, learning_rate=0.02,
                            double_q=True)
                model.learn(total_timesteps=len_episode * num_episode, callback=callback)
                manager.save_model(model)
            else:
                print('il n est pas encore temps')
                classe = Interact(manager=manager, log=1, buffer_size=1000)
                classe.get_tuples_pretrain()
                print(classe.get_tuples_pretrain)

