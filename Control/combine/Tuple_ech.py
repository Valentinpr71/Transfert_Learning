import gym
import pandas as pd
import numpy as np
from buffer_tools.rule_based import rule_based_actions
from buffer_tools.pretrain_policy import Pretrain
from buffer_tools.delete_dict_layer import removeLevel

class Interact():
    """"
    cette classe permet à la fois d'obtenir les tuples d'actions liées aux politiques de contrôle comportementales, qu'elles soient random, basées sur des règles. Puis
    de les utiliser afin de générer les tuples (s_t,a_t,r_t,s_t+1) qui alimenteront le buffer.
    Pour l'argument de l'agent pré-entraîné, on attend un dictionnaire comprenant le nom des fichiers à récupérer.
    """
    def __init__(self, manager, log, buffer_size, replay_buffer, low_noise_p=0.04, rand_action_p=0.3, policy=None):
        self.dim = manager.dim
        self.log = log
        self.agents = manager.parents
        self.buffer_size = buffer_size
        self._get_frac()
        self.len_episode = 8760
        self.data = manager.data
        self.replay_buffer = replay_buffer
        self.low_noise_p = low_noise_p
        self.rand_action_p = rand_action_p
        self.epsilon = 0.01
        self.policy = policy

    def _Action_rule_based(self, epsilon, seed=0):
        Action_rule_based = pd.DataFrame(rule_based_actions(dim=self.dim, data = self.data, epsilon=epsilon, seed=seed))
         #À déplacer, on ne veut pas dupiquer les actions x fois mais directement les tuples
        return(Action_rule_based)

    def _Action_random(self):
        Action_random = pd.DataFrame(np.random.randint(3,size=(8759, int(self.buffer_size*self.frac_random))))
        return(Action_random)

    def _get_frac(self):
        if self.log == 1:
            self.frac_random = 0.6
            self.frac_rule = 0.1
            self.frac_pretrain = 0.7
            self.frac_pretrain_exp = self.frac_pretrain
            self.epsilon = 0.
        elif self.log == 2:
            self.frac_random = 0.2
            self.frac_rule = 0.1
            self.frac_pretrain = 0.2
            self.frac_pretrain_exp = self.frac_pretrain
            self.epsilon = 0.1
        elif self.log == 3:
            self.frac_random = 0.
            self.frac_rule = 0.
            self.frac_pretrain = 0.05
            self.frac_pretrain_exp = self.frac_pretrain
            self.epsilon = 0.3
        else:
            return("ERROR log argument not in range")

    def get_tuples_random(self):
        Action_random=self._Action_random()
        print('len de Action_random : ', len(Action_random))
        env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        rewards_random = {}
        states_random = {}
        actions_random = {}
        for j in range(len(Action_random.columns.value_counts())):
            i = 0
            state_ini = env.reset()
            is_done = False
            rewards_random[j] = {}
            states_random[j] = {0: state_ini}
            actions_random[j] = {0: Action_random[j][0]}
            while not is_done:
                state, reward, is_done, info = env.step(Action_random[j][i])
                i += 1
                actions_random[j][i-1] = Action_random[j][i-1]
                rewards_random[j][i-1] = reward
                states_random[j][i] = state
        tuples_random = {'state': states_random, 'action': actions_random, 'reward': rewards_random}
        return tuples_random

    def get_tuples_rule(self, epsilon_rule):
        env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        duplications = int((self.buffer_size * self.frac_rule)/self.len_episode)
        for j in range(duplications):
            Action_rule_based = self._Action_rule_based(epsilon_rule, seed=j)
            Action_rule_based = Action_rule_based[0]
            i = 0
            obs = env.reset()
            is_done = False
            actions_rule = {0: Action_rule_based[0]}
            episode_start = True
            while not is_done:
                state = obs
                obs, reward, is_done, info = env.step(Action_rule_based[i])
                i += 1
                actions_rule[i-1] = Action_rule_based[i-1]
                action = actions_rule[i-1]
                self.replay_buffer.add(state, action, obs, reward, 0, is_done, episode_start)
                episode_start = False




    def get_tuples_pretrain(self, nb_episodes_per_agent = 0):
        tuples = Pretrain(self.agents,data=self.data, low_noise_p=self.low_noise_p, rand_action_p=self.rand_action_p, replay_buffer=self.replay_buffer, policy=self.policy)
        self.self_dim = self.dim
        tuples.dim = self.self_dim
        replay_buffer = tuples.iterate_parents(nb_episodes_per_agent=nb_episodes_per_agent,epsilon=self.epsilon)
        return replay_buffer

    def build_buffer(self):
        # if self.frac_random:
        #     tuples_random = self.get_tuples_random()
        if self.frac_rule:
            self.get_tuples_rule(epsilon_rule=0.05)
        if self.frac_pretrain:
            print("NB EP PER AGENT : ", int(self.buffer_size*self.frac_pretrain_exp/len(self.agents)))
            self.get_tuples_pretrain(nb_episodes_per_agent=int(self.buffer_size*self.frac_pretrain_exp/(len(self.agents)*self.len_episode)))
            # tuples_pretrain_pretrain = removeLevel(tuples_pretrain, 1)
            # tuples_pretrain_exploration = removeLevel(tuples_pretrain, 1)
