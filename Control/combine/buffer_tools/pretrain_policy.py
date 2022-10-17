import gym
import pandas as pd
import numpy as np
import glob



class Pretrain():
    """
    Cette classe récupère les modèles entraînés demandés et génères des actions dans l'environnement spécifié. C'est un outil pour la classe Tuple_ech.
    Classe à changer si le modèle change de DQN.
    Arguments :
    - dicto : dictionnaire numéroté contenant le nom des fichiers contenant les agents entraînés. AUTRE MANIÈRE plus intéressante: la clé est le hashkey et la valeur le dimensionnement. le dimensionnement doit être un np.array
    - dim : dimension de l'environnement sur lequel les predictions doivent être faites. De la forme d'une liste de flottants
    """
    def __init__(self, dicto, replay_buffer, data=[], low_noise_p=0.01, rand_action_p=0.3, policy = None):
        self.dicto=dicto
        self.dim=None
        self.model=None
        self.data=data
        self.replay_buffer=replay_buffer
        self.low_noise_p = low_noise_p
        self.rand_action_p = rand_action_p
        self.policy0 = policy[0]
        self.policy1 = policy[1]

    def _load_agent(self,filename):
        setting = f"MicrogridControlGym-v0_"+filename
        self.env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        ## modifs pour intégrer pytorch au load de l'agent pour faire comme avec le train_behavioral de BCQ
        self.policy.load(f"./models/{setting}")
        # self.model =
        # self.model = DQN.load("Batch_RL_results/"+filename+"/best_model.zip", env=self.env)

    def tuples_pretrain(self, num_episode, epsilon):
        count=0
        for i in range(num_episode):
            low_noise_ep = np.random.uniform(0, 1) < self.low_noise_p
            count+=low_noise_ep
        print("COUNT : ", count)
        self.pretrain_high_noise(nb_episodes_per_agent=num_episode-count)
        self.pretrain_low_noise(nb_episode=count)



    def pretrain_low_noise(self, nb_episode=0):
        #Déclanche le remplissage du buffer pour le nombre d'épisodes avec low_noise, donc une exploration faible (proba espilon)
        for i in range(nb_episode):
            j = 0
            print('ET DE UN EPISODE EN low NOISE')
            obs = self.env.reset()
            episode_start = True
            is_done = False
            while not is_done:
                #modifié le 8 sept pour adapter a bcq
                action = self.policy.select_action(np.array(obs))
                states = obs
                obs, reward, is_done, info = self.env.step(action)
                # rewards_episode[j-1] = reward
                self.replay_buffer.add(states, action, obs, reward, float(is_done), is_done, episode_start)
                episode_start = False

    def pretrain_high_noise(self, nb_episodes_per_agent):
        #Déclanche le remplissage du buffer avec une plus grande probabilité d'actions aléatoires.
        # states = {}
        # actions = {}
        # rewards = {}
        is_done = False
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
                if np.random.rand()<self.low_noise_p:
                    action = self.env.action_space.sample()
                    # action = np.random.randint(3)
                else:
                    action = self.policy.select_action(np.array(obs))
                    # action, _states = self.model.predict(obs)
                states = obs
                obs, reward, is_done, info = self.env.step(action)
                # rewards_episode[j-1] = reward
                self.replay_buffer.add(states, action, obs, reward, float(is_done), is_done, episode_start)
                episode_start = False


    def iterate_parents(self, nb_episodes_per_agent=0, epsilon=0.01):
        print("PRETRAIN POLICY --> iterate parents, sample tuples from parents policy for self.dicto = ", self.dicto)
        for i in self.dicto:
            self.dim = self.dicto[i]
            print("dim : ",self.dim)
            ### VP : Cette partie un peu tricky permet de charger la "forme" de la politique adaptée selon si elle vient d'un BCQ ou d'un individu qui a intéragit
            # Cette lecture se fait facilement avec le nom du fichier de résultat lié au hashkey qui va permettre de charger la politique. Soit il y a BCQ dans le nom, soit il n'y est pas
            file = glob.glob(f"./results/*{i}.npy")
            if "BCQ" in file[0]:
                self.policy = self.policy1
            else:
                self.policy = self.policy0
            self._load_agent(str(i))
            self.tuples_pretrain(nb_episodes_per_agent, epsilon)
        return self.replay_buffer
