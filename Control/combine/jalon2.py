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
import argparse
parser = argparse.ArgumentParser(description='Les arguments sont les dimensions sur lesquelles analyser et évaluer la politique d un agent déjà entraîné')
parser.add_argument("--dimPV", const=12., nargs='?', default = 12.)
parser.add_argument("--dimbatt", const=15., nargs='?', default = 15.)
parser.add_argument("--plot", help="À l'utilisateur de choisir ce qu'il désire plot: H2 pour le stockage d'H2, batt pour l'énergie stockée dans la batterie, conso pour la consommation, prod pour la production PV et net pour la demande nette", const="", nargs='?', default="")
parser.add_argument("--multiplot", const=False, nargs='?', default=False)
args = parser.parse_args()
#Ce script a pour but d'analyser la politique d'un agent déjà entraîné sur son environnemnt (dimensionnement de microgrid). Pour que cela fonctionne, il faut que les dossier de politique
# de l'agent en question (se terminant par _Q et _optimizer) spoient présents dans le dossier ".models" du projet.

def plot_H2(info,dim):
    label_x = 'Date'
    label_y_H2 = "Energy stockée sous forme d'H2"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_H2)
    plt.title(f"Réserve d'hydrogène en fonction du temps durant le test de la politique sur le dimensionnement {dim} (kWh)")
    plt.plot(df['Date'],info)
    plt.show()

def plot_batt_SOC(batt_SOC,dim):
    batt_ener = list(x*manager.dim[1] for x in batt_SOC)
    label_x = 'Date'
    label_y_SOC = "Energy stockée dans la batterie"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8759, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"SOC de la batterie électro-chimique en fonction du temps durant le test de la politique sur le dimensionnement {dim} (kWh)")
    plt.plot(df['Date'], batt_ener)
    plt.show()

def plot_conso(conso):
    label_x = 'Date'
    label_y_SOC = "Demande de consommation"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"Demande de consommation en fonction du temps (kW)")
    plt.plot(df['Date'], conso)
    plt.show()


def plot_conso_daily_permonth(conso):
    days_per_month2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    label_x = 'Date'
    label_y_PV = "Energy produite par les panneaux PV"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    conso_daily = []
    #somme par jour
    for i in range(356):
        conso_daily = np.append(conso_daily, np.sum(conso[i*24:(i * 24) + (24)]))
    # net=np.append(net,net[-1])
    conso_monthly = []
    for i in range(len(days_per_month2010)-1):
        conso_monthly = np.append(conso_monthly, np.mean(conso_daily[sum(days_per_month2010[0:i]):sum(days_per_month2010[0:i+1])]))
    conso_monthly = np.append(conso_monthly, conso_monthly[-1])
    conso_lin = []
    for i in range(len(conso_monthly)):
        conso_lin = np.append(conso_lin,np.linspace(conso_monthly[i-1],conso_monthly[i],days_per_month2010[i]*24))
    print("conso_month : ", len(conso_monthly))
    plt.xlabel(label_x)
    plt.ylabel(label_y_PV)
    plt.title(f"Consommation (moyenne quotidienne) par mois (kW)")
    plt.plot(df['Date'], conso_lin)
    plt.show()

def plot_prodPV(prodPV, dim):
    label_x = 'Date'
    label_y_SOC = "Energy stockée dans la batterie"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"Production PV lors d'un épisode sur le dimensionnement {dim} (kW)")
    plt.plot(df['Date'], prodPV)
    plt.show()

def plot_prodPV_daily_permonth(prodPV, dim):
    days_per_month2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    label_x = 'Date'
    label_y_PV = "Energy produite par les panneaux PV"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    prod_daily = []
    #somme par jour
    for i in range(356):
        prod_daily = np.append(prod_daily, np.sum(prodPV[i*24:(i * 24) + (24)]))
    # net=np.append(net,net[-1])
    prod_monthly = []
    for i in range(len(days_per_month2010)-1):
        prod_monthly = np.append(prod_monthly, np.mean(prod_daily[sum(days_per_month2010[0:i]):sum(days_per_month2010[0:i+1])]))
    prod_monthly = np.append(prod_monthly, prod_monthly[-1])
    prod_lin = []
    for i in range(len(prod_monthly)):
        prod_lin = np.append(prod_lin,np.linspace(prod_monthly[i-1],prod_monthly[i],days_per_month2010[i]*24))
    plt.xlabel(label_x)
    plt.ylabel(label_y_PV)
    plt.title(f"Production PV (moyenne quotidienne) par mois sur le dimensionnement {dim} (kW)")
    plt.plot(df['Date'], prod_lin)
    plt.show()

def plot_net_demand(net_demand, dim):
    label_x = 'Date'
    label_y_SOC = "Energy stockée dans la batterie"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"Demande nette (demande - production PV) lors d'un épisode sur le dimensionnement {dim} (kWh)")
    plt.plot(df['Date'], net_demand)
    plt.show()

def plot_net_demand_daily_per_month(net_demand, dim):
    days_per_month2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    label_x = 'Date'
    label_y_SOC = "Energy stockée dans la batterie"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    net = []
    #somme par jour
    for i in range(356):
        net = np.append(net, np.sum(net_demand[i*24:(i * 24) + (24)]))
    # net=np.append(net,net[-1])
    net_monthly = []
    for i in range(len(days_per_month2010)-1):
        net_monthly = np.append(net_monthly, np.mean(net[sum(days_per_month2010[0:i]):sum(days_per_month2010[0:i+1])]))
    net_monthly = np.append(net_monthly, net_monthly[-1])
    net_demand_lin = []
    for i in range(len(net_monthly)):
        net_demand_lin = np.append(net_demand_lin,np.linspace(net_monthly[i-1],net_monthly[i],days_per_month2010[i]*24))
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"Demande nette (demande - production PV) moyenne quotidienne sur un mois pour le dimensionnement {dim} (kWh)")
    plt.plot(df['Date'], net_demand_lin)
    plt.show()

def daily_boxplot_net_daily_permonth(net_demand, dim):
    days_per_month2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    label_x = 'Date (Mois)'
    label_y_SOC = "Demande journalière cumulée (kWh)"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    net = []
    # somme par jour
    for i in range(356):
        net = np.append(net, np.sum(net_demand[i * 24:(i * 24) + (24)]))
    # net=np.append(net,net[-1])
    net_monthly = []
    for i in range(len(days_per_month2010)):
        net_monthly.append(net[sum(days_per_month2010[0:i]):sum(days_per_month2010[0:i + 1])])

    # net_monthly = np.append(net_monthly, net_monthly[-1])
    # net_demand_lin = []
    # for i in range(len(net_monthly)):
    #     net_demand_lin = np.append(net_demand_lin,
    #                                np.linspace(net_monthly[i - 1], net_monthly[i], days_per_month2010[i] * 24))
    tick = ["Juin","Juil","Aout","Sept","Oct","Nov","Dec","Jan","Fev","Mars","Avril","Mai"]
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(
        f"Demande nette (demande - production PV) moyenne quotidienne sur un mois pour le dimensionnement {dim} (kWh)")
    plt.boxplot(net_monthly)
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12],tick)
    plt.show()

def plot_net_demand_lin(net_demand, dim):
    days_per_month2010 = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    label_x = 'Date'
    label_y_SOC = "Energy stockée dans la batterie"
    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    net = []
    for i in range(12):
        net = np.append(net, np.sum(net_demand[i * (24 * 30):(i * 24 * 30) + (24 * 30)]))
    net=np.append(net,net[-1])
    net_demand_lin = []
    for i in range(len(net)-1):
        net_demand_lin = np.append(net_demand_lin,np.linspace(net[i-1],net[i],days_per_month2010[i]*24))
    plt.xlabel(label_x)
    plt.ylabel(label_y_SOC)
    plt.title(f"Demande nette (demande - production PV) lors d'un épisode sur le dimensionnement {dim} (kWh)")
    plt.plot(df['Date'], net_demand_lin)
    plt.show()

def multiplots(info, batt_SOC, net_demand, dim):

    df = pd.DataFrame()
    df['Date'] = pd.date_range('2010-01-01', periods=8760, freq='H').values
    days_per_month2010 = [31,28,31,30,31,30,31,31,30,31,30,31]
# Traitement du niveau de charge de la batterie
    batt_ener = list(x * manager.dim[1] for x in batt_SOC)
    batt_ener.append(batt_ener[-1])
# Traitement de la demande nette
    net = []
    for i in range(12):
        net = np.append(net, np.sum(net_demand[i * (24 * 30):(i * 24 * 30) + (24 * 30)]))
    net=np.append(net,net[0])
    net_demand_lin = []
    for i in range(len(net)-1):
        net_demand_lin = np.append(net_demand_lin,np.linspace(net[i-1],net[i],days_per_month2010[i]*24))
    fig, host = plt.subplots(figsize=(8,5))
    # par1 = host.twinx()
    par2 = host.twinx()

    host.set_xlabel("Date")
    host.set_ylabel("Energy stockée sous forme d'H2 (kWh)")
    # par1.set_ylabel("Energy stockée dans la batterie (kWh)")
    par2.set_ylabel(f"Demande nette (demande - production PV) lors d'un épisode sur le dimensionnement {dim} (kWh)")

    color1 = plt.cm.viridis(0)
    # color2 = plt.cm.viridis(0.5)
    color3 = plt.cm.viridis(.9)

    p1, = host.plot(df['Date'], info, color=color1, label="Energy stockée sous forme d'H2 (kWh)")
    # p2, = par1.plot(df['Date'], batt_ener, color=color2, label="Energy stockée dans la batterie (kWh)")
    p3, = par2.plot(df['Date'], net_demand_lin, color=color3, label=f"Demande nette (demande - production PV) lors d'un épisode sur le dimensionnement {dim} (kWh)")

    lns = [p1, p3]# p2, p3]
    host.legend(handles=lns, loc='best')

    # right, left, top, bottom
    par2.spines['right'].set_position(('outward', 60))

    # no x-ticks
    # par2.xaxis.set_ticks([])

    host.yaxis.label.set_color(p1.get_color())
    # par1.yaxis.label.set_color(p2.get_color())
    par2.yaxis.label.set_color(p3.get_color())

    fig.tight_layout()
    plt.show()

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
        cheat_dim_list = [np.array([float(args.dimPV),
                                    float(args.dimbatt)])]  # ,np.array([11.0,11.0]),np.array([11.0, 10.0]),np.array([12.0,12.0]),np.array([11.0,9.0])] #Ajouté uniquement dans un but d'analyse de l'algo au début (on force le nb de voisin a etre petit)
        # dim = np.array([float(np.random.randint(dim_boundaries['PV']['low'], dim_boundaries['PV']['high'])),
        #                 float(np.random.randint(dim_boundaries['batt']['low'], dim_boundaries['batt']['high']))])
        dim = cheat_dim_list[i]
        print("DIM : ",dim)
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
    obs, production, consumption= env.reset()
    done = False
    while not done:
        state = obs
        action = tuples.eval_policy(state)
        tuple_list['state'].append(state[0])
        #tuple_list['H2_storage'].append(tuple_list['H2_storage'][-1]-1)
        obs, reward, done, info = env.step(action)
        tuple_list['reward'].append(reward)
    # my_dict = {'1': 'aaa', '2': 'bbb', '3': 'ccc'}
    # with open('test.csv', 'w') as f:
    #     for key in tuple_list.keys():
    #         f.write("%s,%s"%(key,tuple_list[key]))
    if "H2" in args.plot:
        plot_H2(info,manager.dim)
    if "batt" in args.plot:
        plot_batt_SOC(tuple_list['state'],manager.dim)
    Tableur = env.render_to_file
    if "conso" in args.plot:
        plot_conso(consumption)
    if "daily_conso_permonth" in args.plot:
        plot_conso_daily_permonth(consumption)
    if "prodPV" in args.plot:
        plot_prodPV(production,manager.dim)
    if "net1" in args.plot:
        plot_net_demand_lin(consumption-production, manager.dim)
    if "daily_net_permonth" in args.plot:
        plot_net_demand_daily_per_month(consumption-production, manager.dim)
    if "daily_PV_permonth" in args.plot:
        plot_prodPV_daily_permonth(production, manager.dim)
    if 'boxplot' in args.plot:
        daily_boxplot_net_daily_permonth(consumption-production,manager.dim)
    if args.multiplot:
        multiplots(info, tuple_list['state'], consumption-production, manager.dim)
    print(manager.dim)

