import gym
import pandas as pd
import numpy as np
from stable_baselines import DQN


class Pretrain():
    """
    Cette classe récupère les modèles entraînés demandés et génères des actions dans l'environnement spécifié. C'est un outil pour la classe Tuple_ech.
    Classe à changer si le modèle change de DQN.
    Arguments :
    - dicto : dictionnaire numéroté contenant le nom des fichiers contenant les agents entraînés. AUTRE MANIÈRE plus intéressante: la clé est le hashkey et la valeur le dimensionnement. le dimensionnement doit être un np.array
    - dim : dimension de l'environnement sur lequel les predictions doivent être faites. De la forme d'une liste de flottants
    """
    def __init__(self, dicto, data=[]):
        self.dicto=dicto
        self.dim=None
        self.model=None
        self.data=data

    def _load_agent(self,filename):
        self.env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        self.model = DQN.load("Batch_RL_results/"+filename+"/best_model.zip", env=self.env)

    def tuples_pretrain(self):
        i = 1
        obs = self.env.reset()
        is_done = False
        actions = {}
        states = {0: obs}
        rewards = {}
        while not is_done:
            i += 1
            action, _states = self.model.predict(obs)
            actions[i-1] = action
            #actions=np.append(actions, action)
            obs, reward, is_done, info = self.env.step(action)
            rewards[i-1] = reward
            states[i] = obs
        tuples = {'state': states, 'action': actions, 'reward': rewards}
        return actions, states, rewards

    def pretrain_with_exploration(self,epsilon):
        i = 1
        obs = self.env.reset()
        is_done = False
        actions = {}
        states = {0: obs}
        rewards = {}
        while not is_done:
            if np.random.rand()<epsilon:
                action = np.random.randint(3)
            else:
                action, _states = self.model.predict(obs)
            actions[i-1] = action
            #actions=np.append(actions, action)
            obs, reward, is_done, info = self.env.step(action)
            rewards[i-1] = reward
            states[i] = _states
            i += 1
        tuples = {'state': states, 'action': actions, 'reward': rewards}
        return tuples

    def iterate_parents(self):
        j = 0
        tuples = {}
        actions ={}
        states = {}
        rewards = {}
        for i in self.dicto:
            self.dim = self.dicto[i]
            print("dim : ",self.dim)
            self._load_agent(i)
            (actions[j], states[j], rewards[j]) = self.tuples_pretrain()
            tuples = {'state': states, 'action': actions, 'reward': rewards}
            j += 1
        return tuples
