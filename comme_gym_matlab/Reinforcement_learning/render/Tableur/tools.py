import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd


class Tool(object):
    def __init__(self, env):
        self.env=env
        self.smooth=0
        self.LCE=""
        self.excess=""
        self.ppc=""
        self.no_excess=False
    def name_argument(self):
        if "sell" in self.env:
            self.sell="_sell"
        else:
            self.sell="_no_sell"
        if "lce" in self.env:
            self.LCE="_LCE"
        if "excess" in self.env:
            self.excess="_excess"
        if "ppc" in self.env:
            self.ppc="_ppc"
    def read_data(self):
        df=pd.read_csv("render"+self.LCE+self.excess+self.sell+self.ppc+".csv")
        df=df.drop(['Energy Sold'], axis=1)
        df = df.drop(['Hydrogen storage'], axis=1)
        if self.LCE == "_LCE" and self.sell == "_sell" and self.ppc == "":
            df=df.drop(['Excess Energy'], axis=1)
            print(df)
            self.no_excess=True
        return df
    def _smooth(self,y_data, num):
        box = np.ones(num)/num
        y_smooth = np.convolve(y_data, box, mode='same')
        return y_smooth

    def correlation_matrix(self, correlation):
        rcParams.update({'figure.autolayout': True})
        f = plt.figure(figsize=(14, 14))

        # plt.subplots_adjust(top=5)
        plt.matshow(correlation.corr(), fignum=f.number)
        plt.xticks(range(correlation.select_dtypes(['number']).shape[1]), correlation.select_dtypes(['number']).columns,
                   fontsize=9, rotation=45)
        plt.yticks(range(correlation.select_dtypes(['number']).shape[1]), correlation.select_dtypes(['number']).columns,
                   fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        if self.LCE == "_LCE":
            LCE = "-operating cost "
        else:
            LCE = ""
        if self.excess == '_excess':
            excess = "-excess energy "
        else:
            excess = ''
        if self.sell == "_no_sell":
            case = "Case 1, "
        if self.sell == "_sell" and self.ppc == "_ppc":
            case = "Case 3, "
        if self.sell == "_sell" and self.ppc == "":
            case = "Case 2, "
        plt.title('Correlation Matrix for ' + case + "Reward: " + LCE + excess, fontsize=16)
        plt.savefig("Images/corr/render"+self.LCE+self.excess+self.sell+self.ppc+".png", bbox_inches='tight')
        plt.show()

    def plot_data(self, df):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots()
        ax.set_xlabel("Step in the last episode")
        x=df['step']
        # color="tab:red"
        # ax1.plot(x,self._smooth(df['Reward'], 100), color=color)
        # ax2=ax1.twinx()
        # color="tab:blue"
        # ax2.plot(x,self._smooth(df['Net demand'], 100), color=color)
        color="tab:green"
        #ax.plot(x,self._smooth(df['Hydrogen storage'], 100),color=color, label="Hydrogen Storage")

        color="tab:red"
        ax.plot(x,self._smooth(df['Energy Bought'], 100), color=color, label="Energy Bought")
        if not self.no_excess:
            color = "tab:blue"
            ax.plot(x,self._smooth(df['Excess Energy'], 100), color=color, label="Excess Energy")
        if self.LCE == "_LCE":
            LCE="-operating cost "
        else:
            LCE=""
        if self.excess =='_excess':
            excess="-excess energy "
        else:
            excess=''
        if self.sell =="_no_sell":
            case="Case 1, "
        if self.sell == "_sell" and self.ppc=="_ppc":
            case="Case 3, "
        if self.sell=="_sell" and self.ppc =="":
            case= "Case 2, "
        ax.legend()
        plt.title('Indicators for '+case+"Reward: "+LCE+excess)
        plt.savefig("Images/plot_result/"+self.LCE+self.excess+self.sell+self.ppc+".png", bbox_inches='tight')
        plt.show()