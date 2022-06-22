import gym
import pandas as pd
import numpy as np

class Interact():
    """"
    cette classe permet à la fois d'obtenir les tuples d'actions liées aux politiques de contrôle comportementales, qu'elles soient random, basées sur des règles. Puis
    de les utiliser afin de générer les tuples (s_t,a_t,r_t,s_t+1) qui alimenteront le buffer.
    """
    def __init__(self, dim, agent, log,buffer_size):
        self.dim = dim
        self.log=log
        self.agent=agent
        self.buffer_size=buffer_size
        self.frac_rule=None
        self.frac_random=None
        self.frac_pretrain=None
        self.len_episode=8760
    def _Action_rule_based(self):
        return(Action_rule_based)
    def _Action_random(self):
        Action_random=pd.DataFrame(np.random.randint(3,size=(8760, int(self.buffer_size*self.frac_random))))
        return(Action_random)
    def _get_frac(self):
        if self.log==1:
            self.frac_random=0.4
            self.frac_rule=0.1
            self.frac_pretrain=0.5
        if self.log==2:
            self.frac_random=0.6
            self.frac_rule=0.1
            self.frac_pretrain=0.3
        else:
            return("ERROR log argument not in range")
    def _get_tuples_random(self):
        Action_random=self._Action_random()
        env = gym.make("microgrid:MicrogridControlGym-v0", dim=self.dim)
        rewards_random={}
        states_random={}
        actions_random={}
        for j in range(len(Action_random.columns.value_counts())):
            i=0
            state_ini = env.reset()
            is_done = False
            rewards_random[j]={0:0}
            states_random[j]={0:state_ini}
            actions_random[j]={0:Action_random[j][0]}
            while not is_done:
                state, reward, is_done, info = env.step(Action_random[j][i])
                i += 1
                actions_random[j][i] = Action_random[j][i]
                rewards_random[j][i] = reward
                states_random[j][i] = state
        tuples_random={state}