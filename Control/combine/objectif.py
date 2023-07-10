from discrete_BCQ.main_iter import main_BCQ
from microgrid.envs.import_data import Import_data
from buffer_tools.hash_manager import Dim_manager
from buffer_tools.battery2 import Battery
import pandas as pd
import numpy as np
from os.path import exists
import glob
import torch

class main_dim():
    def __init__(self, distance, nb_voisins):
        self.dicto = {}
        self.manager = Dim_manager(distance=distance, nb_voisins=nb_voisins)


    def is_know_in_section(self, dimPV, dimbatt):
        import_data = Import_data(mode="train")
        consumption_norm, consumption = import_data.data_cons_AutoCalSOl()
        self.manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)
        dim = np.array([float(dimPV), float(dimbatt)])
        self.manager._dim(dim.tolist())
        self.manager.choose_parents()
        return self.manager.add_to_dicto()

    def iterations_dim(self, dimPV, dimbatt):
        trained_already = 0
        self.batt = Battery(type='non-linear', dim = dimbatt)
        dim_boundaries = {'PV': {'low': 0, 'high': 17}, 'batt': {'low': 0, 'high': 15}}
        import_data = Import_data(mode="train")
        # consumption = import_data.consumption()
        # consumption_norm = import_data._consumption_norm()
        # consumption_norm, consumption = import_data.data_cons_AutoCalSOl()
        # VP 01/04/2023
        consumption_norm, consumption, production_norm, production = import_data.split_years(5, 17, '2010_2020_SARAH2.csv',
                                                                                periods=96432, start_time='2010-01-01',
                                                                                years=[2010,2011,2012,2013,2014])
        self.manager.add_data_cons(data_cons=consumption, data_cons_norm=consumption_norm)

        ### Arguments ne variant pas pour le main BCQ:
        env = "MicrogridControlGym-v0"
        seed = 1
        max_timestep = 10e6  # Nombre d'iteration si generate buffer ou train_behavioral. C'est le nombre de tuples utilisés
        buffer_name = "Essai_0" #préfixe au nom du fichier
        BCQ_threshold = 0.3 #tau expliqué sur le README pour Discrete BCQ
        low_noise_p = 0.1 #probabilité que l'épisode soit avec une faible exploration epsilon, dans le cas contraire, il prendra une décision aléatoire avec un taux égal à l'argument rand_action_p
        rand_action_p = 0.3 #probabilité de prendre une action aléatoire si l'on n'est pas dans un low noise episode
        temps = {"behavioral":np.array([]),"bcq":np.array([])}

        dim = np.array([float(dimPV), float(dimbatt)])
        price_PV = 2600*dim[0]/20 ##-->230€/kWc Pour 6kWc 15500€ 80% de la puissance produite à 25 ans. On imagine une durée à 100% grossièrement à 20 ans https://www.hellowatt.fr/panneaux-solaires-photovoltaiques/combien-coute-installation-photovoltaique
        price_batt = 600*dim[1]/8 #--> 75€/kWh ## 6*100€/kWh https://www.powertechsystems.eu/fr/home/technique/etude-de-cout-du-lithium-ion-vs-batteries-au-plomb/ POUR 8 ans (3000 cycles)
        self.manager._dim(dim.tolist())
        self.manager.choose_parents()
        # Noel 2020, voir commentaire plus bas.
        # if self.manager.add_to_dicto():
        # print(f"/results/{env}_{self.manager._create_hashkey()}")
        self.manager.add_to_dicto()
        results = glob.glob("./tests_optim/results/*.npy")
        print("actual hashkey : ", self.manager._create_hashkey())
        if True in ["MicrogridControlGym-v0_"+self.manager._create_hashkey() in i for i in results]:
            print('Agent already trained for this microgrid dimension')
            trained_already = 1
            # rewards = np.load(glob.glob(f"./tests_optim/results/*{self.manager._create_hashkey()}.npy")[0])
            # return np.sum(rewards)
        if True in ["BCQ_MicrogridControlGym-v0_"+self.manager._create_hashkey() in i for i in results]:
        # if exists("./results/"+env+"_"+manager._create_hashkey()+".npy"):
            print('BCQ Agent already trained for this microgrid dimension')
            trained_already = 2
            # rewards = np.load(glob.glob(f"./tests_optim/results/*{self.manager._create_hashkey()}.npy")[0])
            # return np.sum(rewards)
        # production_norm = import_data._production_norm(PV=dim[0])
        # production = import_data.production()
        # production_norm, production = import_data.data_prod_AutoCalSol(dimPV=dim[0], dimPVmax=dim_boundaries['PV']['high'])
        #cons_norm, cons, production_norm, production = import_data.treat_csv(dim[0], dim_boundaries['PV']['high'], "clean_PV_GIS_2018_SARAH2.csv")
        cons_norm, cons, production_norm, production = import_data.split_years(5, 17, '2010_2020_SARAH2.csv',
                                                                                periods=96432, start_time='2010-01-01',
                                                                                years=[2010,2011,2012,2013,2014])
        # cons_norm_test, cons_test, production_norm_test, production_test = import_data.split_years(5, 17, 'PV_GIS_2005_2020_TOULOUSE_SARAH2.csv',
        #                                                                         periods=140256, start_time='2010-01-01',
        #                                                                         years=[2010, 2011, 2012])
        manager_test = self.manager
        self.manager.add_data_prod(data_prod=production, data_prod_norm=production_norm)
        # manager_test.add_data_prod(data_prod=production_test, data_prod_norm=production_norm_test)
        ### Initialisation de l'env, prend 0seconde
        BCQ = main_BCQ(env=env, manager=[self.manager, manager_test], seed=seed, buffer_name=buffer_name, max_timestep=max_timestep,
                       BCQ_threshold=BCQ_threshold, low_noise_p=low_noise_p, rand_action_p=rand_action_p, already_trained=trained_already, battery=self.batt)
        score, temps, tau_autoprod, tau_autocons = BCQ.Iterate(temps)
        obj = (score*0.2*1e-3) + price_PV + price_batt
        # print("TEMPS DE CALCUL : ", temps)
        return obj, score, tau_autoprod, tau_autocons
        #Noel : On enlève la partie if self.ad_to_dicto car l'agent est déjà entraîné sur la session si on ne l'ajoute pas au dict des dim connus de la session. On peut donc utiliser les already_trained.

        # else:
        #     rewards = np.load(glob.glob(f"./tests_optim/results/*{self.manager._create_hashkey()}.npy")[0])
        #     return np.sum(rewards)