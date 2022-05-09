import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Results(object):
    def __init__(self):
        self.date=[]
    def import_data(self):
        self.df1=pd.read_csv("render_LCE_no_sell.csv")
        self.date = pd.date_range('2010-01-01', periods=len(self.df1), freq='H').values
        self.df1['Date']=self.date
        df2 = pd.read_csv("render_LCE_sell.csv")
        self.df3 = pd.read_csv("render_LCE_sell_ppc.csv")
        self.df3['Date'] = self.date
        self.df4 = pd.read_csv('render_excess_no_sell.csv')
        self.df4['Date'] = self.date
        self.df5=pd.read_csv('render_excess_sell_ppc.csv')
        self.df5['Date'] = self.date
        self.df2=df2.drop(['Excess Energy'], axis=1)
        self.df6=pd.read_csv('render_LCE_excess_sell_ppc.csv')
        self.df7=pd.read_csv('render_LCE_excess_no_sell.csv')

    def sum_data(self):
        self.df1["Sum Excess"]=self.df1['Excess Energy'].cumsum()
        self.df3["Sum Excess"]=self.df3['Excess Energy'].cumsum()
        self.df4["Sum Excess"]=self.df4['Excess Energy'].cumsum()
        self.df5["Sum Excess"]=self.df5['Excess Energy'].cumsum()
        self.df6["Sum Excess"]=self.df6['Excess Energy'].cumsum()
        self.df7["Sum Excess"]=self.df7['Excess Energy'].cumsum()
        self.df1["Sum Energy Bought"]=self.df1['Energy Bought'].cumsum()
        self.df2["Sum Energy Bought"] = self.df2['Energy Bought'].cumsum()
        self.df3["Sum Energy Bought"]=self.df3['Energy Bought'].cumsum()
        self.df4["Sum Energy Bought"]=self.df4['Energy Bought'].cumsum()
        self.df5["Sum Energy Bought"]=self.df5['Energy Bought'].cumsum()
        self.df6["Sum Energy Bought"]=self.df6['Energy Bought'].cumsum()
        self.df7["Sum Energy Bought"]=self.df7['Energy Bought'].cumsum()

    def plot_sum_excess(self, case=1):
        fig, ax = plt.subplots()
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulated potential PV energy lost (Wh)")
        x = self.df1.Date
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        ax.set_ylim(
            min(self.df1['Sum Excess'].min(), self.df3['Sum Excess'].min(), self.df4['Sum Excess'].min(),
                self.df5['Sum Excess'].min()), self.df1["Sum Excess"].max())
        if case==1:
            color="tab:red"
            ax.plot(x,self._smooth(self.df1['Sum Excess'], 80), color=color, label="Case 1, reward: -operating cost")
            # ax2=ax1.twinx()
            # ax.set_ylim(self.df1['Excess Energy'].min(), 12)
            #Careful, df2 has no 'Excess Energy'
            # ax4=ax1.twinx()
            # ax4.set_ylim(self.df1['Excess Energy'].min(), 12)
            color="tab:blue"
            ax.plot(x,self._smooth(self.df4["Sum Excess"],80),color=color, label="Case 1, reward: -excess energy")
            color='tab:olive'
            ax.plot(x,self._smooth(self.df7["Sum Excess"],80),color=color, label="Case 1, reward: -operating cost -excess energy")
        if case == 3:
            color="tab:olive"
            ax.plot(x,self._smooth(self.df6["Sum Excess"],80),color=color, label="Case 3, reward: -operating cost -excess energy")
            color='tab:blue'
            ax.plot(x,self._smooth(self.df5["Sum Excess"],80),color=color, label="Case 3, reward: -excess energy")
            # ax3=ax1.twinx()
            # ax3.set_ylim(self.df1['Excess Energy'].min(), 12)
            color="tab:red"
            ax.plot(x,self._smooth(self.df3["Sum Excess"],80),color=color, label="Case 3, reward: -operating cost")
        ax.set_title("Excess energy during the last training episode")
        ax.legend()
        # ax2.legend(loc=2)
        # ax3.legend(loc=3)
        # ax4.legend(loc=4)
        plt.show()

    def plot_sum_energy_bought(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulated energy bought (Wh)")
        x = self.df1.Date
        color = "tab:red"
        ax.set_ylim(
            min(self.df1['Energy Bought'].min(), self.df3['Energy Bought'].min(), self.df4['Energy Bought'].min(),
                self.df5['Energy Bought'].min()), self.df4["Sum Energy Bought"].max())
        ax.plot(x, self._smooth(self.df1['Sum Energy Bought'], 50), color=color,
                label="Case 1, reward: -operating cost")
        color='yellow'
        # Careful, df2 has no 'Excess Energy'
        ax.plot(x, self._smooth(self.df2["Sum Energy Bought"], 50), color=color,
                label="Case 2, reward: -operating cost")
        color = 'tab:blue'
        ax.plot(x, self._smooth(self.df5["Sum Energy Bought"], 50), color=color,
                label="Case 3, reward: -excess energy")
        # ax3=ax1.twinx()
        # ax3.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:green"
        ax.plot(x, self._smooth(self.df3["Sum Energy Bought"], 50), color=color,
                label="Case 3, reward: -operating cost")
        # ax4=ax1.twinx()
        # ax4.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:orange"
        ax.plot(x, self._smooth(self.df4["Sum Energy Bought"], 50), color=color, label="Case 1, reward: -excess energy")
        color="tab:brown"
        ax.plot(x,self._smooth(self.df6["Sum Energy Bought"],50),color=color, label="Case 3, reward: -operating cost -excess energy")
        color='tab:olive'
        ax.plot(x,self._smooth(self.df7["Sum Energy Bought"],50),color=color, label="Case 1, reward: -operating cost -excess energy")
        ax.set_title("Energy Bought during the last training episode")
        ax.legend()
        # ax2.legend(loc=2)
        # ax3.legend(loc=3)
        # ax4.legend(loc=4)
        plt.show()


    def plot_excess(self, case = 1):
        fig, ax = plt.subplots()
        ax.set_xlabel("Date")
        x = self.df1.Date
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        ax.set_ylim(
            min(self.df1['Excess Energy'].min(), self.df3['Excess Energy'].min(), self.df4['Excess Energy'].min(),
                self.df5['Excess Energy'].min()), 11)
        if case==1:
            color="tab:red"
            ax.plot(x,self._smooth(self.df1['Excess Energy'], 80), color=color, label="Case 1, reward: -operating cost")
            # ax2=ax1.twinx()
            # ax.set_ylim(self.df1['Excess Energy'].min(), 12)
            #Careful, df2 has no 'Excess Energy'
            # ax4=ax1.twinx()
            # ax4.set_ylim(self.df1['Excess Energy'].min(), 12)
            color="tab:orange"
            ax.plot(x,self._smooth(self.df4["Excess Energy"],80),color=color, label="Case 1, reward: -excess energy")
            color='tab:olive'
            ax.plot(x,self._smooth(self.df7["Excess Energy"],80),color=color, label="Case 1, reward: -operating cost -excess energy")
        if case == 3:
            color="tab:brown"
            ax.plot(x,self._smooth(self.df6["Excess Energy"],80),color=color, label="Case 3, reward: -operating cost -excess energy")
            color='tab:blue'
            ax.plot(x,self._smooth(self.df5["Excess Energy"],80),color=color, label="Case 3, reward: -excess energy")
            # ax3=ax1.twinx()
            # ax3.set_ylim(self.df1['Excess Energy'].min(), 12)
            color="tab:green"
            ax.plot(x,self._smooth(self.df3["Excess Energy"],80),color=color, label="Case 3, reward: -operating cost")
        ax.set_title("Excess energy during the last training episode")
        ax.legend()
        # ax2.legend(loc=2)
        # ax3.legend(loc=3)
        # ax4.legend(loc=4)
        plt.show()

    def plot_hydrogen(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        fig, ax = plt.subplots()
        ax.set_xlabel("Date")
        ax.set_ylabel('Hydrogen stored (kWh)')
        x = self.df1.Date
        color = "tab:red"
        ax.set_ylim(
            min(self.df1['Hydrogen storage'].min(), self.df3['Hydrogen storage'].min(), self.df4['Hydrogen storage'].min(),
                self.df5['Hydrogen storage'].min()), max(self.df1['Hydrogen storage'].max(), self.df3['Hydrogen storage'].max(), self.df4['Hydrogen storage'].max(),
                self.df5['Hydrogen storage'].max()))
        ax.plot(x, self._smooth(self.df1['Hydrogen storage'], 50), color=color,
                label="Case 1, reward: -operating cost")
        color='yellow'
        # Careful, df2 has no 'Excess Energy'
        ax.plot(x, self._smooth(self.df2["Hydrogen storage"], 50), color=color,
                label="Case 2, reward: -operating cost")
        color = 'tab:blue'
        ax.plot(x, self._smooth(self.df5["Hydrogen storage"], 50), color=color,
                label="Case 3, reward: -excess energy")
        # ax3=ax1.twinx()
        # ax3.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:green"
        ax.plot(x, self._smooth(self.df3["Hydrogen storage"], 50), color=color,
                label="Case 3, reward: -operating cost")
        # ax4=ax1.twinx()
        # ax4.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:orange"
        ax.plot(x, self._smooth(self.df4["Hydrogen storage"], 50), color=color, label="Case 1, reward: -excess energy")
        color="tab:brown"
        ax.plot(x,self._smooth(self.df6["Hydrogen storage"],50),color=color, label="Case 3, reward: -operating cost -excess energy")
        color='tab:olive'
        ax.plot(x,self._smooth(self.df7["Hydrogen storage"],50),color=color, label="Case 1, reward: -operating cost -excess energy")
        ax.set_title("Hydrogen storage during the last training episode")
        ax.legend()
        # ax2.legend(loc=2)
        # ax3.legend(loc=3)
        # ax4.legend(loc=4)
        plt.show()

    def plot_energy_bought(self):
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()
        ax.set_xlabel("Date")

        x = self.df1.Date
        color = "tab:red"
        ax.set_ylim(
            min(self.df1['Energy Bought'].min(), self.df3['Energy Bought'].min(), self.df4['Energy Bought'].min(),
                self.df5['Energy Bought'].min()), 2)
        ax.plot(x, self._smooth(self.df1['Energy Bought'], 50), color=color,
                label="Case 1, reward: -operating cost")
        color='yellow'
        # Careful, df2 has no 'Excess Energy'
        ax.plot(x, self._smooth(self.df2["Energy Bought"], 50), color=color,
                label="Case 2, reward: -operating cost")
        color = 'tab:blue'
        ax.plot(x, self._smooth(self.df5["Energy Bought"], 50), color=color,
                label="Case 3, reward: -excess energy")
        # ax3=ax1.twinx()
        # ax3.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:green"
        ax.plot(x, self._smooth(self.df3["Energy Bought"], 50), color=color,
                label="Case 3, reward: -operating cost")
        # ax4=ax1.twinx()
        # ax4.set_ylim(self.df1['Excess Energy'].min(), 12)
        color = "tab:orange"
        ax.plot(x, self._smooth(self.df4["Energy Bought"], 50), color=color, label="Case 1, reward: -excess energy")
        color="tab:brown"
        ax.plot(x,self._smooth(self.df6["Energy Bought"],50),color=color, label="Case 3, reward: -operating cost -excess energy")
        color='tab:olive'
        ax.plot(x,self._smooth(self.df7["Energy Bought"],50),color=color, label="Case 1, reward: -operating cost -excess energy")
        ax.set_title("Energy Bought during the last training episode")
        ax.legend()
        # ax2.legend(loc=2)
        # ax3.legend(loc=3)
        # ax4.legend(loc=4)
        plt.show()

    def _smooth(self,y_data, num):
        box = np.ones(num)/num
        y_smooth = np.convolve(y_data, box, mode='same')
        return y_smooth
