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
    def __init__(self, manager, log, buffer_size):
        self.dim = manager.dim
        self.log = log
        self.agents = manager.agents_parents
        self.buffer_size = buffer_size
        self._get_frac()
        self.len_episode=8760
        self.data=manager.data
    def _Action_rule_based(self):
        Action_rule_based = pd.DataFrame(rule_based_actions(dim=self.dim, data = self.data))
         #À déplacer, on ne veut pas dupiquer les actions x fois mais directement les tuples
        return(Action_rule_based)

    def _Action_random(self):
        Action_random = pd.DataFrame(np.random.randint(3,size=(8759, int(self.buffer_size*self.frac_random))))
        return(Action_random)

    def _get_frac(self):
        if self.log == 1:
            self.frac_random = 0.6
            self.frac_rule = 0.1
            self.frac_pretrain = 0.3
            self.frac_pretrain_exp = 0.
            self.epsilon = 0.
        elif self.log == 2:
            self.frac_random = 0.2
            self.frac_rule = 0.1
            self.frac_pretrain = 0.2
            self.frac_pretrain_exp = 0.01
            self.epsilon = 0.1
        elif self.log == 3:
            self.frac_random = 0.
            self.frac_rule = 0.
            self.frac_pretrain = 0.05
            self.frac_pretrain_exp = 0.95
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

    def get_tuples_rule(self):
        Action_rule_based = self._Action_rule_based()
        Action_rule_based = Action_rule_based[0]
        print('len de Action_rule_based : ', len(Action_rule_based))
        env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim, data=self.data)
        rewards_rule = {}
        states_rule = {}
        actions_rule = {}
        # for j in range(len(Action_rule_based.columns.value_counts())):
        i = 0
        state_ini = env.reset()
        is_done = False
        rewards_rule = {}
        states_rule = {0: state_ini}
        actions_rule = {0: Action_rule_based[0]}
        while not is_done:
            state, reward, is_done, info = env.step(Action_rule_based[i])
            i += 1
            actions_rule[i-1] = Action_rule_based[i-1]
            rewards_rule[i-1] = reward
            states_rule[i] = state
        states = {}
        actions = {}
        rewards = {}
        duplications = int(self.buffer_size * self.frac_rule)
        for i in range(duplications):
            states[i] = states_rule
            actions[i] = actions_rule
            rewards[i] = rewards_rule
        tuples_rule = {'state': states, 'action': actions, 'reward': rewards}
        #
        # tuples_rule = pd.concat([tuples_rule] * duplication, axis=1, ignore_index=True)
        return tuples_rule

    def get_tuples_pretrain(self, is_explo = False, nb_episodes_per_agent = 0):
        tuples = Pretrain(self.agents,data=self.data)
        tuples.dim = self.dim
        tuples, tuples_explo = tuples.iterate_parents(is_explo=is_explo, nb_episodes_per_agent=nb_episodes_per_agent,epsilon=self.epsilon)
        # tuples_explo = removeLevel(tuples_explo, 1)
        duplication=int(self.buffer_size*self.frac_pretrain/len(self.agents))
        intermediaire = {"state":{},"action":{},"reward":{}}
        for i in tuples['state'].keys():
            for j in range(duplication):
                intermediaire["state"][i*duplication+j] = tuples['state'][i]
                intermediaire["action"][i * duplication + j] = tuples['action'][i]
                intermediaire["reward"][i * duplication + j] = tuples['reward'][i]
        #tuples_pretrain = pd.concat([tuples]*duplication,axis=1,ignore_index=True)
        #print("len tuples_pretrain : ", len(tuples_pretrain), "tuples_pretrain : ", tuples_pretrain)
        return intermediaire, tuples_explo

    def build_buffer(self):
        if self.frac_random:
            tuples_random = self.get_tuples_random()
        if self.frac_rule:
            tuples_rule = self.get_tuples_rule()
        if self.frac_pretrain:
            tuples_pretrain, tuples_pretrain_exploration = self.get_tuples_pretrain(is_explo=(self.frac_pretrain_exp > 0), nb_episodes_per_agent=int(self.buffer_size*self.frac_pretrain_exp/len(self.agents)))
            # tuples_pretrain_pretrain = removeLevel(tuples_pretrain, 1)
            # tuples_pretrain_exploration = removeLevel(tuples_pretrain, 1)

