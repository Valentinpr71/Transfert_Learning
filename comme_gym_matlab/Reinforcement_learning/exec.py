import agent_gym
import os

os.system("python agent_gym.py")
os.system("python agent_gym.py --hydrogene_storage_penality False")
os.system("python agent_gym.py --sell_to_grid")
os.system("python agent_gym.py --sell_to_grid --ppc")
os.system("python agent_gym.py --sell_to_grid --ppc --independancy")
os.system("python agent_gym.py --sell_to_grid --independancy")