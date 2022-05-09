import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.dates as mdates
import pandas as pd

class Test_plot():
    def __init__(self, info, num_episode, hydrogene_storage_penality, ppc, sell_grid, independancy):
        self.hydrogene_storage_penality=""
        self.ppc=""
        self.sell_grid=""
        self.independancy=""
        if hydrogene_storage_penality:
            self.hydrogene_storage_penality=" Hydrogene Storage Penality "
        if ppc !=0:
            if ppc is None:
                self.ppc=" no PPC cons"
            else:
                self.ppc=" PPC max cons= "+str(self.ppc)
        if sell_grid:
            self.sell_grid=" Excess energy goes to main grid "
        if independancy==-1:
            self.independancy=" Grid dependancy penality "
        self.info=info
        self.num_episode=num_episode
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def plot_rewards(self):
        df=pd.DataFrame()
        df['Date'] = pd.date_range('2010-01-01', periods=len(self.info['Timestep']), freq='H').values
        df['Reward']=self.info['Reward'].copy()
        plt.title('Smoothed rewards for episode ')
        plt.suptitle(self.hydrogene_storage_penality+self.ppc+self.sell_grid+self.independancy)
        ax=plt.gca()
        xfmt = mdates.DateFormatter('%Y-%m-%d %H')
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=25)
        print(df['Date'], df['Reward'])
        plt.plot(df['Date'], df['Reward'])
        plt.plot(df['Date'], self.smooth(df['Reward'],3),'r')
        plt.plot(df['Date'], self.smooth(df['Reward'],13),'g')
        # plt.show()

    def save(self):
        fig=plt.figure(figsize=(10,6))

        plt.text(x=0.5, y=0.85, s='Episode '+str(self.num_episode)+'\n'+self.hydrogene_storage_penality+self.ppc+self.sell_grid+self.independancy, fontsize=18, ha="center", transform=fig.transFigure)
        ax1=fig.add_subplot(211)
        ax1.set_title('Smoothed rewards')

        ax2=fig.add_subplot(212)
        ax2.set_title("Energy Surplus")

        df=pd.DataFrame()
        df['Date'] = pd.date_range('2010-01-01', periods=len(self.info['Timestep']), freq='H').values
        df['Reward']=self.info['Reward'].copy()
        df["Surplus"]=self.info["Energy Surplus"]
        # plt.title('Smoothed rewards for episode '+str(self.num_episode))
        # ax=plt.gca()
        xfmt = mdates.DateFormatter('%Y-%m-%d %H')
        # ax.xaxis.set_major_formatter(xfmt)
        # plt.xticks(rotation=25)
        plt.sca(ax1)
        # plt.xaxis.set_major_formatter(xfmt)

        plt.xticks(rotation=25)
        plt.subplots_adjust(top=0.8)
        fig.tight_layout()
        ax1.plot(df['Date'], self.smooth(df['Reward'], 17))

        ax2.plot(df['Date'],self.smooth(df["Surplus"],15))

        plt.savefig('Smoothed Rewards and Surplus for episode '+ str(self.num_episode)+'\n'+self.hydrogene_storage_penality+self.ppc+self.sell_grid+self.independancy)

    def test(self):
        print(self.info, self.num_episode)




# if __name__ =='__main__':
#     info={"Timestep":[],"Reward":[]}
#     for i in range(8760):
#         info['Timestep'].append(i+1)
#         info['Reward'].append(np.random.random())
#     df=pd.DataFrame()
#     df['Date'] = pd.date_range('2010-01-01', periods=len(info['Timestep']), freq='H').values
#     df['ts']=df.Date.astype('int64')
#     df['Reward']=info['Reward'].copy()
#     df['B']=np.arange(0,8760)
#     plot_rewards(df)

if __name__=="__main__":
    info = {"Timestep": [], "Reward": []}
    for i in range(8760):
        info['Timestep'].append(i + 1)
        info['Reward'].append(np.random.random())
    num_episode = 3
    plot = Test_plot(info, num_episode)
    plot.plot_rewards()
    plot.save()

