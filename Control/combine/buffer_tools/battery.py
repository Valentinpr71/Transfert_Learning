
from scipy import integrate as intg

class Battery():
    def __init__(self, type = 'linear', dim = 1, eta = 0.95):
        type=type
        if type == 'linear':
            self.batt = Linear_battery(dim=dim, eta=eta)
        elif type == 'non-linear':
            self.batt = Non_linear_battery(dim=dim, eta=eta)
        else:
            print("Error: Unknown battery type")
    def discharge(self, delta):
        SOC, energy_bought = self.batt._discharge(delta)
        return SOC, energy_bought
    def charge(self,delta):
        SOC, energy_sold = self.batt._charge(delta)
        return SOC, energy_sold

class Linear_battery():
    def __init__(self, dim=1, eta=0.95):
        self.SOC = 0.
        self.Cmax = dim*1000
        self.C = self.Cmax
        self.eta = eta
        self.cycle_perte = 0.000066667/2 # Perte par cycle en pourcentage de Cmax


    def _discharge(self, delta):
        DOD = 1-self.SOC
        if self.SOC*self.Cmax*self.eta>=delta:
            #If there is enough stored energy, use it
            self.SOC = self.SOC-delta/(self.Cmax*self.eta)
            energy_bought = 0
        else:
            #else use what is available and buy the remaining energy
            energy_restored = self.SOC*self.Cmax*self.eta
            self.SOC = 0.
            energy_bought = delta-energy_restored
        deg = self.cycle_perte*(DOD-(1-self.SOC))
        self.C = self.C - self.C*deg/100
        return self.SOC, energy_bought

    def _charge(self, delta):
        DOD=1-self.SOC
        if min(self.SOC*self.C, self.SOC-(delta/self.Cmax)*self.eta)==self.SOC*self.C:
            #Excess energy considering maximum capacity with degradation of the battery
            energy_sold = -delta-(1.*self.C-((self.SOC*self.Cmax)/self.eta))
        else:
            energy_sold = 0
        self.SOC = min(self.SOC*self.C,self.SOC-((delta/self.Cmax)*self.eta))
        deg = self.cycle_perte * (DOD - (1 - self.SOC))
        self.C = self.C - self.C * deg / 100
        return self.SOC, energy_sold



class Non_linear_battery():
    def __int__(self, dim=1, eta=0.95):
        self.SOC = 0.
        self.Cmax = dim*1000
        self.eta = eta
        self.C = self.Cmax
        self.A = 4.83 * 10 ** (-4)
        self.B = 2.38 * 10 ** (-5)

    def deg_function(self, x):
        return (self.A*x**2)+self.B*x

    def _discharge(self, delta):
        DOD = 1-self.SOC
        if self.SOC*self.Cmax*self.eta>=delta:
            #If there is enough stored energy, use it
            self.SOC = self.SOC-delta/(self.Cmax*self.eta)
            energy_bought = 0
        else:
            #else use what is available and buy the remaining energy
            energy_restored = self.SOC*self.Cmax*self.eta
            self.SOC = 0.
            energy_bought = delta-energy_restored
        deg = intg.quad(self.deg_function(), DOD, 1-self.SOC)[0]
        self.C = self.C - self.C*deg/100
        return self.SOC, energy_bought

    def _charge(self, delta):
        if min(self.SOC*self.C, self.SOC-(delta/self.Cmax)*self.eta)==self.SOC*self.C:
            #Excess energy considering maximum capacity with degradation of the battery
            energy_sold = -delta-(1.*self.C-((self.SOC*self.Cmax)/self.eta))
        else:
            energy_sold = 0
        self.SOC = min(self.SOC*self.C,self.SOC-((delta/self.Cmax)*self.eta))
        return self.SOC, energy_sold