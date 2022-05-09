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
    env = args.env
    Tool = Tool(env)
    Tool.name_argument()
    df = Tool.read_data()
    df1 = df[int((len(df) / 6) * 2):int((len(df) / 6) * 4)]
    df1['step'] = df1.iloc[:, 0]
    Tool.plot_data(df1)
    df1.set_index(df1['step'])
    df_corr = df1.drop(["step"], axis=1)
    df_corr.drop(['Unnamed: 0'], axis=1, inplace=True)
    Tool.correlation_matrix(df_corr)