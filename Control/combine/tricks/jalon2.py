from discrete_BCQ.main import main_BCQ
from microgrid.envs.import_data import Import_data
from buffer_tools.hash_manager import Dim_manager
import pandas as pd
import csv
import numpy as np
from os.path import exists
import glob
import torch
from discrete_BCQ import utils
from buffer_tools.pretrain_policy import Pretrain
from discrete_BCQ import DQN
from discrete_BCQ import discrete_BCQ
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    is_atari = False
    dim_num_iteration = 1
    dim_boundaries = {'PV': {'low': 0, 'high': 12}, 'batt': {'low': 0, 'high': 15}}
    distance_euclidienne = 3
    nb_voisins = 1
    manager = Dim_manager(distance=distance_euclidienne, nb_voisins=nb_voisins)
    import_data = Import_data(mode="train")
    consumption = import_data.consumption()
    consumption_norm = import_data._consumption_norm()
    manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)
    ### Arguments ne variant pas pour le main BCQ:
    env = "MicrogridControlGym-v0"
    seed = 1
    max_timestep = 1e6  # Nombre d'iteration si generate buffer ou train_behavioral. C'est le nombre de tuples utilisés

    for i in range(dim_num_iteration):
        print(f"itération {i} de dimensionnement")
        cheat_dim_list = [np.array([8.0,
                                    9.0])]  # ,np.array([11.0,11.0]),np.array([11.0, 10.0]),np.array([12.0,12.0]),np.array([11.0,9.0])] #Ajouté uniquement dans un but d'analyse de l'algo au début (on force le nb de voisin a etre petit)
        # dim = np.array([float(np.random.randint(dim_boundaries['PV']['low'], dim_boundaries['PV']['high'])),
        #                 float(np.random.randint(dim_boundaries['batt']['low'], dim_boundaries['batt']['high']))])
        dim = cheat_dim_list[i]
        manager._dim(dim.tolist())
        manager.choose_parents()

        # modfis effectives

        manager.add_to_dicto()
        production_norm = import_data._production_norm(PV=dim[0])
        production = import_data.production()
        manager.add_data_prod(data_prod=production, data_prod_norm=production_norm)

    env, state_dim, num_actions = utils.make_env(env, manager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BCQ_threshold = 0.3
    regular_parameters = {
        # multi_env
        "nb_parents": 2,
        "euclidian_dist": 4,
        # Exploration
        ### Modifié pour l'abaisser dans les épisodes en low noise
        "start_timesteps": 8760,  # nombre de step avant de ne plus prendre que des actions aléatoires
        "initial_eps": 0.01,
        "end_eps": 0.1,
        "eps_decay_period": 1,
        # Evaluation
        # "eval_freq": 8759,#Attention c'est en nombre de step et pas en nombre d'épisodes
        "eval_freq": 8760,
        "eval_eps": 0.005,
        # Learning
        "discount": 0.75,
        "buffer_size": 1e6,
        "batch_size": 256,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 1e-3
        },
        "train_freq": 1,
        "polyak_target_update": False,
        "target_update_freq": 70000,
        # tau passé de 0.005 à 0.9
        "tau": 0.005
    }

    parameters = regular_parameters
    writer = None
    policy0 = DQN.DQN(is_atari,
                      num_actions,
                      state_dim,
                      device,
                      parameters["discount"],
                      parameters["optimizer"],
                      parameters["optimizer_parameters"],
                      parameters["polyak_target_update"],
                      parameters["target_update_freq"],
                      parameters["tau"],
                      # parameters["initial_eps"],
                      1,
                      # parameters["end_eps"],
                      0.1,
                      # parameters["eps_decay_period"],
                      600000,
                      parameters["eval_eps"],
                      writer,
                      parameters["train_freq"])
    policy1 = discrete_BCQ.discrete_BCQ(
        is_atari,
        num_actions,
        state_dim,
        device,
        BCQ_threshold,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
        writer,
        parameters['train_freq']
    )
    policy = [policy0, policy1]

    tuples = Pretrain(dicto=manager.dicto, data=manager.data, policy=policy)
    tuples.dim = manager.dicto[manager._create_hashkey()]
    tuples.policy = tuples.policy0
    policy = tuples._load_agent(str(manager._create_hashkey()))
    print("Manager dim : ", manager.dim)
    # eval_env, _, _ = utils.make_env(env, manager)
    tuple_list = {"state":[],"action":[],'reward':[]}
    obs, done = env.reset(), False
    while not done:
        state = obs
        action = tuples.eval_policy(state)
        tuple_list['state'].append(state[0])
        #tuple_list['H2_storage'].append(tuple_list['H2_storage'][-1]-1)
        obs, reward, done, info = env.step(action)
        tuple_list['reward'].append(reward)
    env.render_to_file()
    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}
    # with open('test.csv', 'w') as f:
    #     for key in tuple_list.keys():
    #         f.write("%s,%s"%(key,tuple_list[key]))
    label_x = 'Date'
    label_y_H2 = "Energy stockée sous forme d'H2"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_H2)
    plt.plot(df['Date'],info)
    print(manager.dim)
