from discrete_BCQ import utils
import gym
import numpy as np
import matplotlib.pyplot as plt
from discrete_BCQ.discrete_BCQ import discrete_BCQ
from buffer_tools.hash_manager import Dim_manager





def taux_mensuels(dim):
    regular_parameters = {			# multi_env
                "nb_parents": 2,
                "euclidian_dist": 4,
                # Exploration
                ### Modifié pour l'abaisser dans les épisodes en low noise
                "start_timesteps": 8760, #nombre de step avant de ne plus prendre que des actions aléatoires
                "initial_eps": 0.01,
                "end_eps": 0.1,
                "eps_decay_period": 1,
                # Evaluation
                #"eval_freq": 8759,#Attention c'est en nombre de step et pas en nombre d'épisodes
                "eval_freq": 8760,
                "eval_eps": 0.005,
                # Learning
                "discount": 0.9,
                "buffer_size": 1e6,
                "batch_size": 256,
                "optimizer": "Adam",
                "optimizer_parameters": {
                    "lr": 1e-8
                },
                "train_freq": 1,
                "polyak_target_update": False,
                "target_update_freq": 70000,
                #tau passé de 0.005 à 0.9
                "tau": 0.005
            }

    manager = Dim_manager(4,2)
    manager._dim(dim)
    # manager._dim([4, 12])
    # dim = manager.dim

    from microgrid.envs.import_data import Import_data

    import_data = Import_data(mode='train')
    consumption_norm, consumption, production_norm, production = import_data.treat_csv(dim[0], 17, "clean_PV_GIS_2020_SARAH2.csv")
    #consumption_norm, consumption = import_data.data_cons_AutoCalSOl()
    #production_norm, production = import_data.data_prod_AutoCalSol(dimPV=dim[0], dimPVmax=17)
    #manager.add_data_prod(data_prod=production, data_prod_norm=production_norm)
    #manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)
    data = [consumption, consumption_norm, production, production_norm]
    env = gym.make("microgrid:MicrogridControlGym-v0", dim=dim, data=data)
    manager.data = data
    env, state_dim, num_action = utils.make_env("microgrid:MicrogridControlGym-v0", manager)
    writer = None
    device = "cpu"
    BCQ_threshold = 0.3
    parameters  = regular_parameters
    policy = discrete_BCQ(
        False,
        num_action,
        state_dim,
        device,
        BCQ_threshold,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        0,
        0,
        1,
        0,
        writer,
        parameters['train_freq']
    )
    manager._dim([float(manager.dim[0]),float(manager.dim[1])])
    policy.load(f"./tests_optim/models/MicrogridControlGym-v0_{manager._create_hashkey()}")

    list_of_actions = np.array([])
    rewards = np.array([])
    obs = env.reset()
    is_done=False
    while not is_done:
        action = policy.select_action(np.array(obs))
        states = obs
        obs, reward, is_done, info = env.step(action)
        rewards = np.append(rewards, reward)
        list_of_actions = np.append(list_of_actions,action)
    manager._create_hashkey()
    tab = env.render_to_file()
    grp = tab.groupby(tab.index.month)
    tau_autoprod, tau_autocons = [], []
    for i in range(len(grp)):
        tau_autoprod.append(
            1 - (sum(tab['Energy Bought'][grp.groups[i + 1]]) / sum(tab["Consumption"][grp.groups[i + 1]])))
        tau_autocons.append(
            1 - (sum(tab['Energy Sold'][grp.groups[i + 1]]) / sum(tab["PV production"][grp.groups[i + 1]])))
    return(tau_autoprod, tau_autocons)

