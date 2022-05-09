import gym
import pandas
from gym.utils import seeding
import numpy as np
import copy
from base_classes import Environment


class microgrid_control(Environment):
    metadata = {'render.modes': ['human']}

    def __init__(self,rng):
        """Initialization of the microgrid control environment
        Arguments:
            rng: numpy random generator
        """
        ######  Récupération des données normées et non normés ainsi que les bornes pour les données non-normées de chaque dataset
        ##On vire les validations pour le moment

        #Données de consommation
        self.last_ponctual_observation=[0.,0.,0.]
        self._input_dimensions = [(1,),(1,),(1,)] #Pas d'historique donc seul l'état courant est bservé (pour le 1) et 3 entrées : SOC, cons et prod.
        # Get consumption profile in [0,1]
        self.consumption_train_norm=np.load("./data/example_nondeterminist_cons_train.npy")[0:1*365*24]
        #self.consumption_valid_norm=np.load("../../data/example_nondeterminist_cons_train.npy")[365*24:2*365*24]
        self.consumption_test_norm=np.load("./data/example_nondeterminist_cons_test.npy")[0:1*365*24]
        # Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
        self.consumption_train=self.consumption_train_norm*2.1
        #self.consumption_valid=self.consumption_valid_norm*2.1
        self.consumption_test=self.consumption_test_norm*2.1
        self.min_consumption=min(self.consumption_train)
        self.max_consumption=max(self.consumption_train)


        #Données de production
        self.production_train_norm=np.load("./data/BelgiumPV_prod_train.npy")[0:1*365*24]
        #self.production_valid_norm=np.load("../../data/BelgiumPV_prod_train.npy")[365*24:2*365*24] #determinist best is 110, "nondeterminist" is 124.9
        self.production_test_norm=np.load("./data/BelgiumPV_prod_test.npy")[0:1*365*24] #determinist best is 76, "nondeterminist" is 75.2
        # Scale production profile : 12KWp (60m^2) et en kWh
        self.production_train=self.production_train_norm*12000./1000.
        #self.production_valid=self.production_valid_norm*12000./1000.
        self.production_test=self.production_test_norm*12000/1000

        self.min_production=min(self.production_train)
        self.max_production=max(self.production_train)

        #Dimensionnement des batteries court et long terme
        self.battery_size = 15.
        self.battery_eta = 0.9

        self.hydrogen_max_power = 1.1
        self.hydrogen_eta = .65


    def act(self, action):
        # On calcule la demande nette en enlevant la consommation a la production, directement sur le datatset normé
        true_demand=self.consumption[self.counter-1]-self.production[self.counter-1]
        terminal=0
        if (action==0):
            ## Energy is taken out of the hydrogen reserve
            true_energy_avail_from_hydrogen=-self.hydrogen_max_power*self.hydrogen_eta
            diff_hydrogen=-self.hydrogen_max_power
        if (action==1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen=0
            diff_hydrogen=0
        if (action==2):
            ## Energy is taken into the hydrogen reserve
            true_energy_avail_from_hydrogen=self.hydrogen_max_power/self.hydrogen_eta
            diff_hydrogen=self.hydrogen_max_power

        reward=diff_hydrogen*0.1 # 0.1euro/kWh of hydrogen
        self.hydrogen_storage+=diff_hydrogen

        Energy_needed_from_battery = true_demand + true_energy_avail_from_hydrogen

        if (Energy_needed_from_battery>0):
        # Lack of energy
            if (self._last_ponctual_observation[0]*self.battery_size>Energy_needed_from_battery):
            # If enough energy in the battery, use it
                self._last_ponctual_observation[0]=self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size/self.battery_eta
            else:
            # Otherwise: use what is left and then penalty
                reward-=(Energy_needed_from_battery-self._last_ponctual_observation[0]*self.battery_size)*2 #2euro/kWh
                self._last_ponctual_observation[0]=0


        elif (Energy_needed_from_battery<0):
        # Surplus of energy --> load the battery
            self._last_ponctual_observation[0]=min(1.,self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)


        self._last_ponctual_observation[1]=self.consumption_norm[self.counter]
        self._last_ponctual_observation[2]=self.production_norm[self.counter]

        self.counter+=1
        return copy.copy(reward)

    def reset(self, mode=-1):
        """
        return:  First observations after the reset (begining of an episode)
        """
        self._last_ponctual_observation = [1.,0.,0.]
        self.production_norm = self.production_train_norm
        self.production = self.production_train
        self.consumption_norm = self.consumption_train_norm
        self.consumption = self.consumption_train

        self.counter = 1
        self.hydrogen_storage=0.

        return [0,0,0]

    def inputDimensions(self):
        return self._input_dimensions

    def nActions(self):
        return 3

    def Observe(self):
        return copy.deepcopy(self.last_ponctual_observation)

    def InTerminalState(self):
        return False

    def ObservationType(self):
        return np.float32

    def observe(self):
        return copy.deepcopy(self._last_ponctual_observation)

    def render(self, mode='human', close=False):
        pass
    def end(self):
        pass