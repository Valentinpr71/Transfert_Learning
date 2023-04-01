import gym
from gym import spaces
import numpy as np
import pandas as pd
import datetime
import copy
from .import_data import Import_data
from render.testplot2 import Test_plot


class microgrid_control_gym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, plot_every=10, ppc=None, sell_to_grid=True, total_timesteps=8760 * 10, month=[], dim=[12., 15.],
                 data=[], date_initiale="2022-06-21"):#,date_initiale="2010-06-21"):  # ,month=[9,10,11,12,1,2]):
        print("dim :  eff", dim)
        self._max_episode_steps = 8760
        self.month = month
        self.max_episode = total_timesteps / self._max_episode_steps
        super(microgrid_control_gym, self).__init__()
        self.ppc_power_constraint = ppc
        self.sell_to_grid = sell_to_grid
        if sell_to_grid:
            print("sell to the grid, proof : " + str(self.sell_to_grid), " ppc power constraint : ",
                  self.ppc_power_constraint)
        else:
            self.ppc_power_constraint = 0
            print("no sell to grid, proof : " + str(self.sell_to_grid))

        # self._last_ponctual_observation = [1.,0.,0.]
        # Action space:
        self.action_space = spaces.Discrete(3)
        self.plot_every = plot_every
        # State space:
        # self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0.]), high=np.array([1., 1., 1., 1]))
        self.observation_space = spaces.Box(low=np.array([0., 0., 0., 0., 0.]), high=np.array([1., 1., 1., 1, 1.]))

        #nov2022 : Ajout de l'aspect "Date initiale" de la simulation
        dates2 = pd.date_range(date_initiale, periods=8760, freq='H').values
        datetime3 = pd.DatetimeIndex(dates2)
        year = int(date_initiale[:4])
        gen = (datetime3[i].replace(year=year) for i in range(len(datetime3)))
        ind_new = pd.DatetimeIndex(gen)
        # Définition de la variable observable "distance à l'équinoxe d'été"
        dates = pd.date_range(str(year)+'-01-01', periods=8760, freq='H').values
        datetime2 = pd.DatetimeIndex(dates)
        a = (datetime2 - datetime.datetime(year, 6, 21)).days
        b = (datetime2 - datetime.datetime(year+1, 6, 21)).days
        ab = pd.DataFrame()
        ab['a'] = a
        ab['b'] = b
        self.dist_equinox = ab.abs().min(axis=1)
        self.consumption = data[0]
        self.consumption_norm = data[1]
        self.production = data[2]
        self.production_norm = data[3]
        #nov 2022 : création d'un dataframe afin de renverser l'ordre des données en fonction
        df=pd.DataFrame(index = datetime2)
        df["consumption"] = self.consumption
        df['consumption_norm'] = self.consumption_norm
        df['production'] = self.production
        df['production_norm'] = self.production_norm
        df['dist_equinox'] = np.array(self.dist_equinox)
        df2 = df.reindex(index = ind_new)
        self.dist_equinox = df2["dist_equinox"]
        self.consumption_norm = df2["consumption_norm"]
        self.consumption = df2["consumption"]
        self.production_norm = df2['production_norm']
        self.production = df2["production"]

        if self.month != []:
            consumption = np.array([])
            consumption_norm = np.array([])
            production_norm = np.array([])
            production = np.array([])
            dist_equinox = np.array([])
            for i in self.month:
                consumption = np.concatenate((consumption, self.consumption[datetime2.month == i]), axis=None)
                consumption_norm = np.concatenate((consumption_norm, self.consumption_norm[datetime2.month == i]),
                                                  axis=None)
                production_norm = np.concatenate((production_norm, self.production_norm[datetime2.month == i]),
                                                 axis=None)
                production = np.concatenate((production, self.production[datetime2.month == i]), axis=None)
                dist_equinox = np.concatenate((dist_equinox, self.dist_equinox[datetime2.month == i]), axis=None)
            self.production = production
            self.production_norm = production_norm
            self.consumption = consumption
            self.consumption_norm = consumption_norm
            self.dist_equinox = dist_equinox

        self._max_episode_steps = len(self.consumption_norm)
        print('LONGUEUR D UN EPISODE:', self._max_episode_steps)

        # Dimensionnement des batteries court et long terme
        self.battery_size = dim[1]*1000
        self.battery_eta = 0.9
        self.battery_storage = [0]
        # self.hydrogen_max_power = 2.1*1000
        self.hydrogen_max_power = 1*1000 # PENSER À CHANGER COEFFNORM DANS LE SCRIPT MAIN.PY
        self.hydrogen_elec_eta = .65  # electrolyser eta
        self.hydrogen_PAC_eta = .5  # PAC eta
        self.max_H2_stock = 455000
        ##self.info={"Timestep":[0],"Reward":[0],"Energy Surplus":[0]}
        self.num_episode = 0
        self.info = {}
        ### On a rajouté l'observation de la quantité d'H2 produite à l'agent. Il faut le normer par un nombre
        self.normH2 = 2.1*8760

    def step(self, action):
        # arf=""
        # On calcule la demande nette en enlevant la consommation a la production, directement sur le datatset normé
        true_demand = self.consumption[self.counter - 1] - self.production[self.counter - 1]
        dist_equinox = self.dist_equinox[self.counter - 1]
        reward = 0
        is_done = 0
        if self.counter - len(self.consumption) + 1 == 0:
            is_done = 1
        # Prise en compte des actions
        if action == 0:
            ## Energy is taken out of the hydrogen reserve
            if self.hydrogen_storage[
                -1] - self.hydrogen_max_power < 0:  # If not enough storage to discharge at max power:
                true_energy_avail_from_hydrogen = -self.hydrogen_storage[-1] * self.hydrogen_PAC_eta
                diff_hydrogen = -self.hydrogen_storage[-1]
            else:
                true_energy_avail_from_hydrogen = -self.hydrogen_max_power * self.hydrogen_PAC_eta  # true_energy_avail_from_hydrogen is the delta view from the microgrid system
                diff_hydrogen = -self.hydrogen_max_power  # diff hydrogen is the delta from the hydrogen storage point of view
        if (action == 1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen = 0
            diff_hydrogen = 0
        if (action == 2):
            ## Energy goes into the hydrogen reserve
            true_energy_avail_from_hydrogen = self.hydrogen_max_power
            diff_hydrogen = self.hydrogen_max_power * self.hydrogen_elec_eta
            if self.hydrogen_storage[-1]+diff_hydrogen>self.max_H2_stock:
                #On vérifie si on ne dépasse pas la quantité de H2 max. Si c'est le cas, on charge jusqu'à ce qu'il soit rempli. Sinon, on charge normalement.
                diff_hydrogen=self.max_H2_stock-self.hydrogen_storage[-1]
                #On peut donc effacer (1/etaH2) fois cette puissance.
                true_energy_avail_from_hydrogen = diff_hydrogen*(1/self.hydrogen_elec_eta)

        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen

        if (Energy_needed_from_battery > 0):
            # Lack of energy
            self.energy_sold.append(0)
            if (self._last_ponctual_observation[0] * self.battery_size * self.battery_eta > Energy_needed_from_battery):
                self.energy_bought.append(0)
                # If enough energy in the battery, use it
                self._last_ponctual_observation[0] = self._last_ponctual_observation[0] - Energy_needed_from_battery / (
                        self.battery_size * self.battery_eta)  # battery_eta au dénominateur car on décharge, on doit fournir plus d'éléctricité avec le rendement pris en compte
            else:
                # Otherwise: use what is left and then penalty
                self.energy_bought.append(Energy_needed_from_battery - self._last_ponctual_observation[
                    0] * self.battery_size * self.battery_eta)  # On décharge tout quoiqu'il arrive dans cette situation, donc eta au nominateur pour savoir de combien on a déchargé
                reward -= (Energy_needed_from_battery - self._last_ponctual_observation[
                    0] * self.battery_size * self.battery_eta)
                # On est obligé de pomper dans le réseau central au prix de 2€/kWh
                self._last_ponctual_observation[0] = 0
        elif Energy_needed_from_battery == 0:
            self.energy_bought.append(0)
        elif Energy_needed_from_battery < 0:
            self.energy_bought.append(0)
            # Surplus of energy --> load the battery
            if min(1., self._last_ponctual_observation[0] - (
                    Energy_needed_from_battery / self.battery_size) * self.battery_eta) == 1. and self.sell_to_grid:
                # If the battery is full and there is a surplus of energy, sell it to the grid at 0.5€ per kW. As the (mean of production is 1.5 kW), we define self.ppc_power_constraint as the PCC threshold
                if self.ppc_power_constraint == None:
                    # reward -= (self._last_ponctual_observation[
                    #                0] * self.battery_size - Energy_needed_from_battery * self.battery_eta - 1) * 0.5  # 27 juin : J'ai multiplié par eta plutot que de diviser et rajouté un -1 pour ce qui concerne le surplus d'énergie # octobre 22: pas sûr pour le -1 (analyse dim)

                    #### VP mars 23 : On enlève la récompense ici puisque l'agent ne minimise plus l'énergie vendue au grid, même s'il en vent (osef de l'énergie vendu alors que sell_to_grid =True
                    ####reward -= (-1*Energy_needed_from_battery) - ((1.0*self.battery_size-(self._last_ponctual_observation[0]*self.battery_size))/self.battery_eta) #octobre : R = energy needed -(taille batt -(energie)) pour n'avoir que le surplus on divise par eta car on ne pénalise pas l'énergie perdu dans le rendement de charge, ce n'est pas une énergie vendue
                    # self.energy_sold.append(self._last_ponctual_observation[
                    #                             0] * self.battery_size - Energy_needed_from_battery * self.battery_eta - 1)  # 27 juin : idem
                    #jan 2023 :
                    self.energy_sold.append((-1*Energy_needed_from_battery) - ((1.0*self.battery_size-(self._last_ponctual_observation[0]*self.battery_size))/self.battery_eta))
                else:
                    threshold = min(self.ppc_power_constraint, self._last_ponctual_observation[
                        0] * self.battery_size - Energy_needed_from_battery * self.battery_eta - 1)  # 27 juin : idem
                    if threshold == self.ppc_power_constraint:  # If the surplus>PCC max power
                        reward -= self.ppc_power_constraint #* 0.5
                        self.energy_sold.append(self.ppc_power_constraint)
                    else:
                        reward -= (self._last_ponctual_observation[
                                       0] * self.battery_size - Energy_needed_from_battery * self.battery_eta - 1)
                        self.energy_sold.append(self._last_ponctual_observation[
                                                    0] * self.battery_size - Energy_needed_from_battery * self.battery_eta - 1)  # 27 juin:idem

            elif min(1., self._last_ponctual_observation[
                             0] - Energy_needed_from_battery / self.battery_size * self.battery_eta) == 1. and self.sell_to_grid == False:
                # batterie remplie mais pas de vente au grid
                self.energy_sold.append(0)
            else:  # batterie pas remplie donc pas de surplus ni de vente
                self.energy_sold.append(0)
            self._last_ponctual_observation[0] = min(1., self._last_ponctual_observation[0] - (Energy_needed_from_battery / self.battery_size) * self.battery_eta)
        self.battery_storage.append(self._last_ponctual_observation[0]*self.battery_size)
        self._last_ponctual_observation[1] = self.consumption_norm[self.counter]
        self._last_ponctual_observation[2] = self.production_norm[self.counter]
        self._last_ponctual_observation[3] = self.dist_equinox[self.counter] / 182
        self._last_ponctual_observation[4] = self.hydrogen_storage[-1]/self.max_H2_stock
        # print("counter :", self.counter, "dist equinoxe : ", self.dist_equinox[self.counter],"last obs 0 : ",self._last_ponctual_observation[0], 'reward : ',reward)
        obs = self._last_ponctual_observation
        self.hydrogen_storage.append(self.hydrogen_storage[-1] + diff_hydrogen)
        self.counter += 1
        self.reward = reward
        # jan 2023 :  on coupe render_to_file pour le moment, on le rendra dispo lors de évaluations des politiques pour analyses
        # if self.num_episode-self.max_episode==0.:
        #     self.render_to_file()
        return copy.copy(np.array(obs)), copy.copy(reward), is_done, self.hydrogen_storage
        # return copy.copy(obs), copy.copy(reward), is_done, info

    def reset(self):
        """
        return:  First observations after the reset (begining of an episode)
        """
        self.energy_bought = [0]
        self.energy_sold = [0]
        self.counter = 1
        print("RESET : " + str(self.num_episode))
        self.num_episode += 1
        self.trades = []
        self._last_ponctual_observation = [1., 0., 0., self.dist_equinox[self.counter - 1] / 182, 0.]
        # self._last_ponctual_observation = [1., 0., 0., self.dist_equinox[self.counter - 1] / 182]

        ##self.info = {"Timestep": [0], "Reward": [0],"Energy Surplus":[0]}
        self.battery_storage = [0.]
        self.hydrogen_storage = [0.]
        self.info = {}
        #nov 2022 : on return aussi les données pour pouvoir les visualiser en changeant le point de départ
        # print(np.array([1., 0., 5state0., self.dist_equinox[self.counter - 1] / 182]), self.production, self.consumption)
        # return(np.array([1., 0., 0., self.dist_equinox[self.counter - 1] / 182]))#, self.production, self.consumption)
        return(np.array([1., 0., 0., self.dist_equinox[self.counter - 1] / 182, 0.]))#, self.production, self.consumption)

    def render(self):
        pass

    def save_plot_every(self):
        self.info[0] = self.counter
        self.info[1] = self.reward
        # if self.counter-len(self.consumption)+1==0:
        #     plot=Test_plot(self.info, self.num_episode, hydrogene_storage_penality=self.hydrogen_storage_penalty, ppc=self.ppc_power_constraint, sell_grid=self.sell_to_grid)
        #     plot.save()

    # def render_to_file(self, filename='render.txt'):
    #     self.info[0] = self.counter
    #     self.info[1] = self.reward
    #     if self.counter - self._max_episode_steps == 0:
    #         Tableur = pd.DataFrame()
    #         Tableur['PV production'] = self.production_norm * 12000. / 1000.
    #         Tableur['Consumption'] = self.consumption_norm * 2.1
    #         Tableur['Net demand'] = Tableur['Consumption'] - Tableur['PV production']
    #         # if self.counter>3000:
    #         print('REWARDS : ', self.info[1], "EXCESS : ", self.info[2])
    #         Tableur['Reward'] = self.info[1]
    #         Tableur['Excess Energy'] = self.info[2]
    #         Tableur['Hydrogen storage'] = self.hydrogen_storage
    #         Tableur['Energy Bought'] = self.energy_bought
    #         Tableur['Energy Sold'] = self.energy_sold
    #         Tableur.to_csv('render.csv', sep=',')

    ## Modifs de novembre 2022 pour afficher les éléments dont on a besoin pour une visualisation de ce qu'il se passe dans l'env.
    def render_to_file(self, filename='render.txt'):
        # if self.counter - self._max_episode_steps == 0:
        Tableur = pd.DataFrame()
        Tableur['PV production'] = self.production#self.production_norm * 12000. / 1000.
        Tableur['Consumption'] = self.consumption #self.consumption_norm * 2.1
        Tableur['Net demand'] = self.consumption-self.production#Tableur['Consumption'] - Tableur['PV production']
        # if self.counter>3000:
        #Tableur['Excess Energy'] = self.info[2]
        Tableur['Battery Storage'] = self.battery_storage
        Tableur['Hydrogen_storage'] = self.hydrogen_storage#/self.max_H2_stock
        Tableur['Energy Bought'] = self.energy_bought
        Tableur['Energy Sold'] = self.energy_sold
        # Tableur.to_csv('render.csv', sep=',')
        return Tableur


