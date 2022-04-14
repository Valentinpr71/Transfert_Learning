import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tools import Tool
import argparse
parser = argparse.ArgumentParser(description='Sert à afficher les résultats')
parser.add_argument("--env", help = "The environment to import, can be by default microgrid_control_gymenv2 or if called microgrid_control_gymenv2_excess", nargs='?',default=None)
args = parser.parse_args()
print(args)


if __name__=="__main__":
    # env=args.env
    # Tool=Tool(env)
    #
    # plt.rcParams["figure.figsize"] = [7.50, 3.50]
    # plt.rcParams["figure.autolayout"] = True

    # headers = ['step', 'Net demand', 'Hydrogen storage']
    #
    # df = pd.read_csv('render_excess_no_sell.csv')
    # df['step']=df.iloc[:,0]
    # df.set_index(df['step'])
    # df_corr=df.drop(["step"],axis=1)
    # df_corr.drop(['Unnamed: 0'],axis=1,inplace=True)
    # plt.plot(df['step'],Tool()._smooth(df['Reward'],17))
    #
    # plt.show()
    #
    # Tool().correlation_matrix(df_corr)

    env=args.env
    Tool=Tool(env)
    Tool.name_argument()
    df = Tool.read_data()
    df['step'] = df.iloc[:, 0]
    Tool.plot_data(df)
    df.set_index(df['step'])
    df_corr = df.drop(["step"], axis=1)
    df_corr.drop(['Unnamed: 0'], axis=1, inplace=True)
    Tool.correlation_matrix(df_corr)