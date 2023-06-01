
from scipy import integrate as intg

class Battery():
    def __init__(self, type = 'linear', dim = 1, eta = 0.9):
        self.type=type
        if type == 'linear':
            self.deg = lambda x,y : self.cycle_perte * (abs(x - y))
        elif type == 'non-linear':
            self.deg = lambda x,y : max(0,intg.quad(self.deg_function,x,y)[0])
        elif type=='none':
            self.deg=lambda x,y: 0
        else:
            print("Error: Unknown battery type")
        self.SOC_ini = 1.
        self.SOC = self.SOC_ini
        self.Cmax = dim*1000
        self.eta = eta
        self.C = self.Cmax #Capacité max de la batterie après dégradation
        self.A = 4.83 * 10 ** (-4) ## Coefficients dans la formule de dégradation non-linéaire
        self.B = 2.38 * 10 ** (-5)
        self.cycle_perte = 0.000066667/2 # Perte par cycle en pourcentage de Cmax
        self.histo_Cbatt = [] #Historique de capacité pour mesure l'évolution de la dégradation

    def deg_function(self, x):
        return (self.A*x**2)+self.B*x


    def charge(self, delta):
        DOD=1-self.SOC

        # if min(self.SOC*self.C, self.SOC-(delta/self.C)*self.eta)==self.SOC*self.C:# Ne fonctionne pas car la première partie est bien supérieure à 1 et la seconde proche de 1
        # if min(self.C/self.Cmax, self.SOC-(delta/self.Cmax)*self.eta)==self.C/self.Cmax:
        if min(1., self.SOC-(delta/self.C)*self.eta)==1.:
            #Excess energy considering maximum capacity with degradation of the battery
            # energy_sold = -delta-(1.*self.C-((self.SOC*self.Cmax)/self.eta))
            energy_sold = min(-delta - (self.C - (self.SOC * self.C)),-delta)

        else:
            energy_sold = 0
        # self.SOC = min(self.C / self.Cmax, self.SOC - ((delta / self.Cmax) * self.eta))
        ### Tentative de train avec le SOC calculé en fonction de C et non Cmax:
        self.SOC = min(1., self.SOC - ((delta / self.C) * self.eta))
        self.C = self.C - self.C * self.deg(DOD, (1 - self.SOC))
        self.histo_Cbatt.append(self.C)

        return self.SOC, energy_sold, 1-(self.C/self.Cmax)

    def discharge(self, delta):
        DOD = 1-self.SOC
        if self.SOC*self.C*self.eta>=delta:
        # if self.SOC*self.Cmax*self.eta>=delta:
            #If there is enough stored energy, use it
            self.SOC = self.SOC-delta/(self.C*self.eta)
            energy_bought = 0
        else:
            #else use what is available and buy the remaining energy
            energy_restored = self.SOC*self.C*self.eta
            self.SOC = 0.
            energy_bought = min(delta-energy_restored, delta)
        self.C = self.C - self.C*self.deg(DOD, (1-self.SOC))
        self.histo_Cbatt.append(self.C)
        return self.SOC, energy_bought, 1-(self.C/self.Cmax)

    def reset(self):
        self.SOC=self.SOC_ini
        self.C=self.Cmax
        self.histo_Cbatt = []
    # def discharge(self, delta):
    #     SOC, energy_bought = self.batt._discharge(delta)
    #     return SOC, energy_bought
    # def charge(self,delta):
    #     SOC, energy_sold = self.batt._charge(delta)
    #     return SOC, energy_sold
#
# class Linear_battery():
#     def __init__(self, dim=1, eta=0.95):
#         self.SOC = 0.
#         self.Cmax = dim*1000
#         self.C = self.Cmax
#         self.eta = eta
#         self.cycle_perte = 0.000066667/2 # Perte par cycle en pourcentage de Cmax
#







# class Non_linear_battery():
#     def __int__(self, dim=1, eta=0.95):
#         self.SOC = 0.
#         self.Cmax = dim*1000
#         self.eta = eta
#         self.C = self.Cmax
#         self.A = 4.83 * 10 ** (-4)
#         self.B = 2.38 * 10 ** (-5)
#
#
#
#     def _discharge(self, delta):
#         DOD = 1-self.SOC
#         if self.SOC*self.Cmax*self.eta>=delta:
#             #If there is enough stored energy, use it
#             self.SOC = self.SOC-delta/(self.Cmax*self.eta)
#             energy_bought = 0
#         else:
#             #else use what is available and buy the remaining energy
#             energy_restored = self.SOC*self.Cmax*self.eta
#             self.SOC = 0.
#             energy_bought = delta-energy_restored
#         deg = intg.quad(self.deg_function(), DOD, 1-self.SOC)[0]
#         self.C = self.C - self.C*deg/100
#         return self.SOC, energy_bought
#
#     def _charge(self, delta):
#         if min(self.SOC*self.C, self.SOC-(delta/self.Cmax)*self.eta)==self.SOC*self.C:
#             #Excess energy considering maximum capacity with degradation of the battery
#             energy_sold = -delta-(1.*self.C-((self.SOC*self.Cmax)/self.eta))
#         else:
#             energy_sold = 0
#         self.SOC = min(self.SOC*self.C,self.SOC-((delta/self.Cmax)*self.eta))
#         return self.SOC, energy_sold

