import numpy as np
import sys

class Import_data():
    def __init__(self, mode="train"):
        self.mode=mode
        self.consumption_norm = []
        self.production_norm = []
    def _consumption_norm(self):
        #sys.path.append("/gym/envs/microgridenv")
        print(sys.path)
        self.consumption_norm=np.load("/home/vpere/Gym_Matlab/Escape32/venv/lib/python3.7/site-packages/gym/envs/microgridenv/data/example_nondeterminist_cons_"+self.mode+".npy")[0:1*365*24]
        return self.consumption_norm
    def consumption(self):
        if self.consumption_norm == []:
            self.consumption_norm=self._consumption_norm()
        return self.consumption_norm*2.1
    def _production_norm(self):
        self.production_norm = np.load("/home/vpere/Gym_Matlab/Escape32/venv/lib/python3.7/site-packages/gym/envs/microgridenv/data/BelgiumPV_prod_"+self.mode+".npy")[0:1 * 365 * 24]
        return self.production_norm
    def production(self):
        if self.production_norm == []:
            self.production_norm = self._production_norm()
        return self.production_norm*12000./1000.
    def minmax(self):
        return min(self.consumption()),max(self.consumption()),min(self.production()), max(self.production())