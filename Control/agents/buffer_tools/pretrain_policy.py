import gym
import pandas as pd
import numpy as np
from stable_baselines import DQN


class Pretrain():
    """
    Cette classe récupère les modèles entraînés demandés et génères des actions dans l'environnement spécifié. C'est un outil pour la classe Tuple_ech.
    Classe à changer si le modèle change de DQN.
    Arguments :
    - agent : dictionnaire numéroté contenant le nom des fichiers contenant les agents entraînés. AUTRE MANIÈRE plus intéressante: la clé est le hashkey et la valeur le dimensionnement. le dimensionnement doit être un np.array
    - dim : dimension de l'environnement sur lequel les predictions doivent être faites. De la forme d'une liste de flottants
    """
    def __init__(self,agent, dim):
        self.agent=agent
        self.dim=dim
        self.model=None
    def _load_agent(self,filename):
        self.model =  DQN.load("Result/"+filename)
    def tuples_pretrain(self):
        i=1
        env=gym.make("microgrid:MicrogridControlGym-v0", dim = self.dim)
        obs=env.reset()
        is_done=False
        actions={}
        states={0:obs}
        rewards={}
        while not is_done:
            action, _states=self.model.predict(obs)
            actions[i-1]=action
            #actions=np.append(actions, action)
            obs, reward, is_done = env.step(action)
            rewards[i-1]=reward
            states[i]=_states
            i+=1
        tuples={'state':states,'action':actions,'reward':rewards}
        return tuples