from discrete_BCQ.main import main_BCQ
from microgrid.envs.import_data import Import_data
from buffer_tools.hash_manager import Dim_manager
import pandas as pd
import numpy as np
from os.path import exists
import glob
import torch

if __name__ == "__main__":
    batt='non-linear'
    dim_num_iteration=1
    dim_boundaries = {'PV': {'low': 0, 'high': 17}, 'batt': {'low': 0, 'high': 15}}
    distance_euclidienne = 2
    nb_voisins = 1
    manager = Dim_manager(distance=distance_euclidienne,nb_voisins=nb_voisins)
    import_data = Import_data(mode="train")
    # consumption = import_data.consumption()
    # consumption_norm = import_data._consumption_norm()
    consumption_norm, consumption = import_data.data_cons_AutoCalSOl()
    manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)

    ### Arguments ne variant pas pour le main BCQ:
    env = "MicrogridControlGym-v0"
    seed = 1
    max_timestep = 1e6  # Nombre d'iteration si generate buffer ou train_behavioral. C'est le nombre de tuples utilisés
    buffer_name = "Essai_0" #préfixe au nom du fichier
    BCQ_threshold = 0.3 #tau expliqué sur le README pour Discrete BCQ
    low_noise_p = 0.1 #probabilité que l'épisode soit avec une faible exploration epsilon, dans le cas contraire, il prendra une décision aléatoire avec un taux égal à l'argument rand_action_p
    rand_action_p = 0.3 #probabilité de prendre une action aléatoire si l'on n'est pas dans un low noise episode
    temps = {"behavioral":np.array([]),"bcq":np.array([])}

    for i in range(dim_num_iteration):
        print(f"itération {i} de dimensionnement")
        cheat_dim_list = [np.array([4.,4.]), np.array([3.,3.]),np.array([3.0, 4.0]),np.array([7.0,8.0]),np.array([9.0,5.0])] #Ajouté uniquement dans un but d'analyse de l'algo au début (on force le nb de voisin a etre petit)
        # dim = np.array([float(np.random.randint(dim_boundaries['PV']['low'], dim_boundaries['PV']['high'])),
        #                 float(np.random.randint(dim_boundaries['batt']['low'], dim_boundaries['batt']['high']))])
        dim = cheat_dim_list[i]
        manager._dim(dim.tolist())
        manager.choose_parents()


        if manager.add_to_dicto():
            print(f"/results/{env}_{manager._create_hashkey()}")
            results = glob.glob("./results/*.npy")
            if True in [manager._create_hashkey() in i for i in results]:
            # if exists("./results/"+env+"_"+manager._create_hashkey()+".npy"):
                print('Agent already trained for this microgrid dimension')
                continue
            # production_norm = import_data._production_norm(PV=dim[0])
            # production = import_data.production()
            # production_norm, production = import_data.data_prod_AutoCalSol(dimPV=dim[0], dimPVmax=dim_boundaries['PV']['high'])
            cons_norm, cons, production_norm, production = import_data.treat_csv(dim[0], dim_boundaries['PV']['high'], "clean_PV_GIS_2020_SARAH2.csv")
            manager.add_data_prod(data_prod=production, data_prod_norm=production_norm)
            BCQ = main_BCQ(env=env, manager=manager, seed=seed, buffer_name=buffer_name, max_timestep=max_timestep,
                           BCQ_threshold=BCQ_threshold, low_noise_p=low_noise_p, rand_action_p=rand_action_p)
            temps = BCQ.Iterate(temps)
            print("TEMPS DE CALCUL : ", temps)