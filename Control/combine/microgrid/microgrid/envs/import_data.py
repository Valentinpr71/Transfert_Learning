import numpy as np
import pandas as pd

class Import_data():
    def __init__(self, mode="train"):
        self.mode=mode
        self.consumption_norm = []
        self.production_norm = []
        self.PV=0.

    def _consumption_norm(self):
        self.consumption_norm=np.load("data/example_nondeterminist_cons_"+self.mode+".npy")[0:1*365*24]
        return self.consumption_norm

    def consumption(self):
        if self.consumption_norm == []:
            self.consumption_norm=self._consumption_norm()
        return self.consumption_norm*2.1

    def _production_norm(self, PV=12.):
        self.PV=PV
        self.production_norm = self.PV*(np.load("data/BelgiumPV_prod_"+self.mode+".npy")[0:1 * 365 * 24])/12.
        return self.production_norm

    def production(self):
        if self.production_norm == []:
            self.production_norm = self._production_norm()
        return self.production_norm*self.PV

    def minmax(self):
        return min(self.consumption()),max(self.consumption()),min(self.production()), max(self.production())

    def data_cons_AutoCalSOl(self):
        data_1kW = pd.read_csv("data/Demo_autoconso2.csv")*2
        data_1kW['Dates'] = pd.date_range('2022-01-01', periods=8760, freq='H').values
        data_1kW.set_index(data_1kW['Dates'], inplace=True)
        self.data_1kW = data_1kW
        self.consumption_norm = data_1kW.Consumption / data_1kW.Consumption.max()
        return self.consumption_norm, data_1kW.Consumption

    def data_prod_AutoCalSol(self, dimPV, dimPVmax):
        data_dim = self.data_1kW.copy()
        data_dim_max = self.data_1kW.copy()
        data_dim["ProdPV"] = data_dim["ProdPV"] * dimPV*1.5
        data_dim_max["ProdPV"] = data_dim_max["ProdPV"] * dimPVmax
        self.production_norm = data_dim.ProdPV/data_dim_max.ProdPV.max()
        production = data_dim.ProdPV
        return self.production_norm, production

    def treat_csv(self, dimPV, dimPVmax, filename):
        print('HELLOOWW')
        ##Caution, loaded data have to contain 8760 elements, and columns have to be named "ProdPV" and "Consumption". it has to be 1 kWp solar pannel production data.
        data_1kW = pd.read_csv(f"data/{filename}")
        data_1kW['Dates'] = pd.date_range('2022-01-01', periods=8760, freq='H').values
        data_1kW.set_index(data_1kW['Dates'], inplace=True)
        data_1kW.Consumption *=2
        self.consumption_norm = data_1kW.Consumption / data_1kW.Consumption.max()
        data_dim_max = data_1kW.copy()
        data_dim = data_1kW.copy()
        data_dim["ProdPV"] = data_dim["ProdPV"] * dimPV*1.5
        data_dim_max["ProdPV"] = data_dim_max["ProdPV"] * dimPVmax
        self.production_norm = data_dim.ProdPV/data_dim_max.ProdPV.max()
        return self.consumption_norm, data_1kW.Consumption, self.production_norm, data_dim.ProdPV