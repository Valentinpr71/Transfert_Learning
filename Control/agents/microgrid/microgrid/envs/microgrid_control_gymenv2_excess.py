import gym
from gym import spaces
import numpy as np
import pandas as pd
import copy
from .import_data import Import_data
from render.testplot2 import Test_plot

class microgrid_control_gym2(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, data='train', plot_every=10, ppc=None, sell_to_grid=False,hydrogene_storage_penalty=False, total_timesteps=8760*10):
        self.len_episode = 8760
        self.max_episode=total_timesteps/self.len_episode
        super(microgrid_control_gym2,self).__init__()
        self.ppc_power_constraint=ppc
        self.sell_to_grid=sell_to_grid
        self.excess=0
        if sell_to_grid:
            print("sell to the grid, proof : "+str(self.sell_to_grid), " ppc power constraint : ", self.ppc_power_constraint)
        else:
            self.ppc_power_constraint=0
            print("no sell to grid, proof : "+str(self.sell_to_grid))
        self.hydrogen_storage_penalty=hydrogene_storage_penalty
        self.data=data
        self._last_ponctual_observation = [1.,0.,0.]
        #Action space:
        self.action_space = spaces.Discrete(3)
        self.plot_every=plot_every
        #State space:
        #øelf.observation_space=spaces.Bow(4,)
        self.observation_space =spaces.Box(low=np.array([0.,0.,0.]), high=np.array([1.,1.,1.]))
        # self.observation_space.low=np.array([0.,0.,0.])
        # self.observation_space.high=np.array([1.,1.,1.])

        #Initialisation des données:
        # #Données de consommation
        # self.last_ponctual_observation=[0.,0.,0.]
        # self._input_dimensions = [(1,),(1,),(1,)] #Pas d'historique donc seul l'état courant est bservé (pour le 1) et 3 entrées : SOC, cons et prod.
        # # Get consumption profile in [0,1]
        # self.consumption_train_norm=np.load("../data/example_nondeterminist_cons_train.npy")[0:1*365*24]
        # #self.consumption_valid_norm=np.load("../../data/example_nondeterminist_cons_train.npy")[365*24:2*365*24]
        # self.consumption_test_norm=np.load("../data/example_nondeterminist_cons_test.npy")[0:1*365*24]
        # # Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
        # self.consumption_train=self.consumption_train_norm*2.1
        # #self.consumption_valid=self.consumption_valid_norm*2.1
        # self.consumption_test=self.consumption_test_norm*2.1
        # self.min_consumption=min(self.consumption_train)
        # self.max_consumption=max(self.consumption_train)
        #
        #
        # #Données de production
        # self.production_train_norm=np.load("../data/BelgiumPV_prod_train.npy")[0:1*365*24]
        # #self.production_valid_norm=np.load("../../data/BelgiumPV_prod_train.npy")[365*24:2*365*24] #determinist best is 110, "nondeterminist" is 124.9
        # self.production_test_norm=np.load("../data/BelgiumPV_prod_test.npy")[0:1*365*24] #determinist best is 76, "nondeterminist" is 75.2
        # # Scale production profile : 12KWp (60m^2) et en kWh
        # self.production_train=self.production_train_norm*12000./1000.
        # #self.production_valid=self.production_valid_norm*12000./1000.
        # self.production_test=self.production_test_norm*12000/1000
        #
        # self.min_production=min(self.production_train)
        # self.max_production=max(self.production_train)

        #Consumption
        import_data = Import_data(mode=self.data)
        self.consumption_norm=import_data._consumption_norm()
        self.consumption = import_data.consumption()
        self.production_norm= import_data._production_norm()
        self.production = import_data.production()



        #Dimensionnement des batteries court et long terme
        self.battery_size = 15.
        self.battery_eta = 0.9

        self.hydrogen_max_power = 1.1
        self.hydrogen_elec_eta = .65 #electrolyser eta
        self.hydrogen_PAC_eta =.5 #PAC eta
        self.info={"Timestep":[0],"Reward":[0],"Energy Surplus":[0]}
        self.num_episode=0



    def step(self, action):
        # arf=""

        # On calcule la demande nette en enlevant la consommation a la production, directement sur le datatset normé
        true_demand=self.consumption[self.counter-1]-self.production[self.counter-1]
        reward=0
        is_done=0
        self.excess=0
        if self.counter-len(self.consumption)+1==0:
            is_done=1
        # Prise en compte des actions
        if (action==0):
            ## Energy is taken out of the hydrogen reserve
            if self.hydrogen_storage[-1]-self.hydrogen_max_power<0: #If not enough storage to discharge at max power:
                true_energy_avail_from_hydrogen=-self.hydrogen_storage[-1]*self.hydrogen_PAC_eta
                diff_hydrogen=-self.hydrogen_storage[-1]
            else:
                true_energy_avail_from_hydrogen=-self.hydrogen_max_power*self.hydrogen_PAC_eta #true_energy_avail_from_hydrogen is the delta view from the microgrid system
                diff_hydrogen=-self.hydrogen_max_power #diff hydrogen is the delta from the hydrogen storage point of view
        if (action==1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen=0
            diff_hydrogen=0
        if (action==2):
            ## Energy goes into the hydrogen reserve
            true_energy_avail_from_hydrogen=self.hydrogen_max_power
            diff_hydrogen=self.hydrogen_max_power*self.hydrogen_elec_eta

        # Une récompense est donnée
        if self.hydrogen_storage_penalty:
            reward = -diff_hydrogen * 0.1  # 0.1euro/kWh of hydrogen stored when stored, the opposite if taken out of storage
        # la récompense est négative si l'on pompe dans la réserve de H2, positive si on recharge, nulle sinon


        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen


        if (Energy_needed_from_battery>0):
            self.energy_sold.append(0)
        # Lack of energy
            if (self._last_ponctual_observation[0]*self.battery_size>Energy_needed_from_battery):
                # arf+="a"
                self.energy_bought.append(0)
            # If enough energy in the battery, use it
                self._last_ponctual_observation[0]=self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size/self.battery_eta
            else:
                # arf+="b"
            # Otherwise: use what is left and then penalty
                self.energy_bought.append(Energy_needed_from_battery-self._last_ponctual_observation[0]*self.battery_size)
                reward-=(Energy_needed_from_battery-self._last_ponctual_observation[0]*self.battery_size)*2
            #On est obligé de pomper dans le réseau central au prix de 2€/kWh
                self._last_ponctual_observation[0]=0
            self.info["Energy Surplus"].append(0.)
        elif (Energy_needed_from_battery==0):
            # arf+="v"
            self.info["Energy Surplus"].append(0.)
            self.energy_bought.append(0)
            self.energy_sold.append(0)

        elif (Energy_needed_from_battery<0):
            self.energy_bought.append(0)
        # Surplus of energy --> load the battery
            if min(1.,self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)==1 and self.sell_to_grid:
            #If the battery is full and there is a surplus of energy, sell it to the grid at 0.5€ per kW. As the (mean of production is 1.5 kW), we define self.ppc_power_constraint as the PCC threshold
                if self.ppc_power_constraint==None:
                    # arf+="i"
                    reward += (self._last_ponctual_observation[
                                   0] * self.battery_size - Energy_needed_from_battery / self.battery_eta)*0.5
                    self.energy_sold.append(self._last_ponctual_observation[
                                   0] * self.battery_size - Energy_needed_from_battery / self.battery_eta)
                    self.info["Energy Surplus"].append(self.excess)
                else:
                    threshold = min(self.ppc_power_constraint,self._last_ponctual_observation[0]*self.battery_size-Energy_needed_from_battery/self.battery_eta)
                    if threshold==self.ppc_power_constraint:#If the surplus>PCC max power
                        # arf+="j"
                        self.excess=(self._last_ponctual_observation[0]*self.battery_size-Energy_needed_from_battery/self.battery_eta)-self.ppc_power_constraint
                        self.info['Energy Surplus'].append(self.excess)
                        reward+=self.ppc_power_constraint*0.5
                        self.energy_sold.append(self.ppc_power_constraint)
                    else:
                        # arf+="k"
                        reward+=(self._last_ponctual_observation[0]*self.battery_size-Energy_needed_from_battery/self.battery_eta)*0.5
                        self.info["Energy Surplus"].append(self.excess)
                        self.energy_sold.append(self._last_ponctual_observation[0]*self.battery_size-Energy_needed_from_battery/self.battery_eta)

            elif min(1.,self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)==1 and self.sell_to_grid == False:
                # arf+="d"
                self.excess=self._last_ponctual_observation[0]*self.battery_size-Energy_needed_from_battery/self.battery_eta
                self.info["Energy Surplus"].append(self.excess)
                self.energy_sold.append(0)
            else:
                # arf+="q"
                self.info["Energy Surplus"].append(self.excess)
                self.energy_sold.append(0)
            self._last_ponctual_observation[0]=min(1.,self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)
        # print(arf)
        reward-=self.excess
        self._last_ponctual_observation[1]=self.consumption_norm[self.counter]
        self._last_ponctual_observation[2]=self.production_norm[self.counter]
        obs=self._last_ponctual_observation
        self.hydrogen_storage.append((self.hydrogen_storage[-1]+diff_hydrogen))
        self.counter += 1
        self.reward=reward
        info=self.info
        # if self.num_episode%self.plot_every==0:# or self.num_episode<5:
        #     self.save_plot_every()
        if self.num_episode-self.max_episode==0.:
            self.render_to_file()
        return copy.copy(obs), copy.copy(reward), is_done, info


    def reset(self):
        """
        return:  First observations after the reset (begining of an episode)
        """
        self.energy_bought=[0]
        self.energy_sold=[0]
        self.excess=0
        print(self.num_episode-self.max_episode)
        if self.num_episode!=0:
            print(self.counter-self.len_episode)
        self.counter = 1
        print("RESET : "+str(self.num_episode))
        self.num_episode+=1
        self.trades=[]
        self._last_ponctual_observation = [1.,0.,0.]
        # self.production_norm = self.production_train_norm
        # self.production = self.production_train
        # self.consumption_norm = self.consumption_train_norm
        # self.consumption = self.consumption_train
        self.info = {"Timestep": [0], "Reward": [0],"Energy Surplus":[0]}

        self.hydrogen_storage=[0.]

        return [1.,0.,0.]




    def render(self):
        pass

    def save_plot_every(self):
        self.info["Timestep"].append(self.counter)
        self.info["Reward"].append(self.reward)
        if self.counter-len(self.consumption)+1==0:
            plot=Test_plot(self.info, self.num_episode, hydrogene_storage_penality=self.hydrogen_storage_penalty, ppc=self.ppc_power_constraint, sell_grid=self.sell_to_grid)
            plot.save()


    def render_to_file(self, filename='render.txt'):
        # reward=self.reward
        self.info["Timestep"].append(self.counter)
        self.info["Reward"].append(self.reward)
        if self.counter - self.len_episode == 0:
            Tableur=pd.DataFrame()
            # file = open(filename, 'a+')
            # file.write(f'Step: {self.counter}\n')
            # file.write(f'PV prod: {self._last_ponctual_observation[2]*12000./1000.}\n')

            Tableur['PV production']=self.production_norm*12000./1000.
            Tableur['Consumption']=self.consumption_norm*2.1
            Tableur['Net demand']=Tableur['Consumption']-Tableur['PV production']
            # if self.counter>3000:
            print('REWARDS : ', self.info["Reward"], "EXCESS : ", self.info['Energy Surplus'])
            Tableur['Reward']=self.info["Reward"]
            Tableur['Excess Energy']=self.info['Energy Surplus']
            Tableur['Hydrogen storage']=self.hydrogen_storage
            Tableur['Energy Bought'] = self.energy_bought
            Tableur['Energy Sold'] = self.energy_sold
            # file.write(f'Conso: {self._last_ponctual_observation[1]*2.1}\n')
            # file.write(f'hydrogene storage: {self.hydrogen_storage} (Battery storage: {self._last_ponctual_observation[0]})\n')
            # file.write(f'Net Demand: {self.consumption[self.counter-1]-self.production[self.counter-1]}\n')
            # file.write(f'Excess Energy: {self.info["Energy Surplus"][-1]}\n')
            # file.write(f'Reward: {reward}\n')
            # file.close()
            Tableur.to_csv('render.csv',sep=',')