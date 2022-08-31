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
    def __init__(self, dicto, replay_buffer, data=[]):
        self.dicto=dicto
        self.dim=None
        self.model=None
        self.data=data
        self.replay_buffer=replay_buffer

    def _load_agent(self,filename):
        self.env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        self.model = DQN.load("Batch_RL_results/"+filename+"/best_model.zip", env=self.env)

    def tuples_pretrain(self, low_noise_p, num_episode, high_noise_rate, epsilon):
        count=0
        for i in range(num_episode):
            low_noise_ep = np.random.uniform(0, 1) < low_noise_p
            count+=low_noise_ep
        self.pretrain_high_noise(high_noise_rate=high_noise_rate, nb_episodes_per_agent=num_episode-count)
        self.pretrain_low_noise(epsilon=epsilon, nb_episode=count)

        # i = 1 #correspond au timestep dans un épisode
        # obs = self.env.reset()
        # is_done = False
        # actions = {}
        # states = {}
        # rewards = {}
        # while not is_done:
        #     states[i] = obs
        #     i += 1
        #     action, _states = self.model.predict(obs)
        #     actions[i-1] = action
        #     #actions=np.append(actions, action)
        #     obs, reward, is_done, info = self.env.step(action)
        #     rewards[i-1] = reward
        # return actions, states, rewards

    def pretrain_low_noise(self,epsilon=0.01, nb_episode=0):
        #Déclanche le remplissage du buffer pour le nombre d'épisodes avec low_noise, donc une exploration faible (proba espilon)
        # states = {}
        # actions = {}
        # rewards = {}
        for i in range(nb_episode):
            j = 0
            print('ET DE UN EPISODE EN HIGH NOISE')
            obs = self.env.reset()
            episode_start = True
            is_done = False
            # actions_episode = {}
            # states_episode = {0: obs}
            # rewards_episode = {}
            while not is_done:
                if np.random.rand()<epsilon:
                    action = np.random.randint(3)
                else:
                    action, _states = self.model.predict(obs)
                states = obs
                # j += 1
                # actions_episode[j-1] = action
                #actions=np.append(actions, action)
                obs, reward, is_done, info = self.env.step(action)
                # rewards_episode[j-1] = reward
                self.replay_buffer.add(states, action, obs, reward, 0, is_done, episode_start)
                episode_start = False
        #     states[i] = states_episode
        #     actions[i] = actions_episode
        #     rewards[i] = rewards_episode
        # tuples = {'state': states, 'action': actions, 'reward': rewards}
        # return actions, states, rewards
        # i = 1 #correspond au timestep dans un épisode
        # obs = self.env.reset()
        # is_done = False
        # actions = {}
        # states = {}
        # rewards = {}
        # while not is_done:
        #     states[i] = obs
        #     i += 1
        #     action, _states = self.model.predict(obs)
        #     actions[i-1] = action
        #     #actions=np.append(actions, action)
        #     obs, reward, is_done, info = self.env.step(action)
        #     rewards[i-1] = reward
        # return actions, states, rewards

    def pretrain_high_noise(self,high_noise_rate, nb_episodes_per_agent):
        #Déclanche le remplissage du buffer avec une plus grande probabilité d'actions aléatoires.
        # states = {}
        # actions = {}
        # rewards = {}
        for i in range(nb_episodes_per_agent):
            j = 0
            print('ET DE UN EPISODE EN HIGH NOISE')
            obs = self.env.reset()
            episode_start = True
            is_done = False
            # actions_episode = {}
            # states_episode = {0: obs}
            # rewards_episode = {}
            while not is_done:
                if np.random.rand()<high_noise_rate:
                    action = np.random.randint(3)
                else:
                    action, _states = self.model.predict(obs)
                states = obs
                # j += 1
                # actions_episode[j-1] = action
                #actions=np.append(actions, action)
                obs, reward, is_done, info = self.env.step(action)
                # rewards_episode[j-1] = reward
                self.replay_buffer.add(states, action, obs, reward, 0, is_done, episode_start)
                episode_start = False
        #     states[i] = states_episode
        #     actions[i] = actions_episode
        #     rewards[i] = rewards_episode
        # tuples = {'state': states, 'action': actions, 'reward': rewards}
        # return actions, states, rewards

    def iterate_parents(self, low_noise_p=0.01, nb_episodes_per_agent=0, epsilon=0.01, high_noise_rate=0.35):
        # nb_episode_per_agent est le nombre d'épisodes à faire en explorant par agent.
        j = 0 #correspond au nombre de parents
        # tuples = {}
        # actions ={}
        # states = {}
        # rewards = {}
        # tuples_explo = {}
        # actions_explo ={}
        # states_explo = {}
        # rewards_explo = {}
        for i in self.dicto:
            self.dim = self.dicto[i]
            print("dim : ",self.dim)
            self._load_agent(i)
            # if is_explo:
            #     print('IS EXPLO !!!!!')
            #     (actions_explo[j], states_explo[j], rewards_explo[j]) = self.pretrain_with_exploration(epsilon=epsilon, nb_episodes_per_agent=nb_episodes_per_agent)
            #     tuples_explo = {'state': states_explo, 'action': actions_explo, 'reward': rewards_explo}
            # (actions[j], states[j], rewards[j]) = self.tuples_pretrain(low_noise_p, nb_episodes_per_agent, high_noise_rate, epsilon)
            self.tuples_pretrain(low_noise_p, nb_episodes_per_agent, high_noise_rate, epsilon)
            # tuples = {'state': states, 'action': actions, 'reward': rewards}
            #
            # j += 1
        return self.replay_buffer
