import numpy as np

class Import_data():
    def __init__(self, mode="train",PV=12.):
        self.mode=mode
        self.consumption_norm = []
        self.production_norm = []
        self.PV=PV
    def _consumption_norm(self):
        self.consumption_norm=np.load("data/example_nondeterminist_cons_"+self.mode+".npy")[0:1*365*24]
        return self.consumption_norm
    def consumption(self):
        if self.consumption_norm == []:
            self.consumption_norm=self._consumption_norm()
        return self.consumption_norm*2.1
    def _production_norm(self):
        self.production_norm = self.PV*(np.load("data/BelgiumPV_prod_"+self.mode+".npy")[0:1 * 365 * 24])/12.
        return self.production_norm
    def production(self):
        if self.production_norm == []:
            self.production_norm = self._production_norm()
        return self.production_norm*self.PV
    def minmax(self):
        return min(self.consumption()),max(self.consumption()),min(self.production()), max(self.production())