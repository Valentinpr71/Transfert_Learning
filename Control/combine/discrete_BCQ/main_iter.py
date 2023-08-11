import argparse
import copy
import importlib
import json
import os
import time
from microgrid.envs.import_data import Import_data
import numpy as np
import torch
import pandas as pd
import gym

# from Callbacks.SavingBestRewards import SaveOnBestTrainingRewardCallback
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines.bench import Monitor
from . import discrete_BCQ
from . import DQN
from . import utils
from Tuple_ech import Interact
from torch.utils.tensorboard import SummaryWriter

class main_BCQ():
	def __init__(self, env, manager, seed=0, buffer_name="Default", max_timestep=1e6, BCQ_threshold=0.3, low_noise_p=0.1, rand_action_p=0.3, already_trained=0, battery=None):
		self.writer = SummaryWriter()
		self.already_trained = already_trained
		self.deg_finale_batt = []
		##### VP 17/02 : Je retourne sur l'opti après l'étude en mono_objectif
		self.regular_parameters = {
			"nb_parents": 1,
			"euclidian_dist": 1,
			# Exploration
			### Modifié pour l'abaisser dans les épisodes en low noise
			"start_timesteps": 8760*6, #nombre de step avant de ne plus prendre que des actions aléatoires
			"initial_eps": 0.1,
			"end_eps": 0.001,
			"eps_decay_period": 250e4,
			# "eps_decay_period": 250e4,
			# Evaluation
			"eval_freq": 4, #Attention c'est en nombre de step et pas en nombre d'épisodes
			"eval_eps": 0,
			# Learning
			"discount": 0.999,
			"buffer_size": 6e6,
			"batch_size": 64,
			"optimizer": "Adam",
			"optimizer_parameters": {
				"lr": 1e-3
			},
			"train_freq": 4,
			"polyak_target_update": False,
			"target_update_freq": 100,
			#tau passé de 0.005 à 0.9
			"tau": 0.001
		}
		torch.manual_seed(0)

		# Load parameters
		self.env = env
		self.batt=battery
		self.seed = seed
		self.buffer_name = buffer_name
		self.max_timestep = max_timestep
		self.BCQ_threshold = BCQ_threshold
		self.low_noise_p = low_noise_p
		self.rand_action_p = rand_action_p
		self.manager = manager[0]
		self.manager_test = manager[1]



	if not os.path.exists("./tests_optim/results"):
		print("no folder created yet")
		os.makedirs("./tests_optim/results")
	print("folder created")
	if not os.path.exists("./tests_optim/models"):
		os.makedirs("./tests_optim/models")

	if not os.path.exists("./tests_optim/buffers"):
		os.makedirs("./tests_optim/buffers")

	def interact_with_environment(self, env, replay_buffer, is_atari, device, inter=None):
		# For saving files
		manager = self.manager
		coeff_norm = max((manager.data[0].max()),(manager.data[3].max()))
		setting = f"{self.env}_{list(manager.dicto.keys())[-1]}" #On met le hashkey correspondant au dimensionnement actuel dans le nom. On retrouve ce dimensionnement comme le dernier ajouté au dictionnaire du manager
		buffer_name = f"{self.buffer_name}_{setting}"
		print("SETING : ", setting)
		print("MANAGER : ", manager)
		# Initialize and load policy
		parameters=self.regular_parameters
		policy0 = DQN.DQN(
			is_atari,
			self.num_action,
			self.state_dim,
			device,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			#parameters["initial_eps"],
			1,
			#parameters["end_eps"],
			0.1,
			# parameters["eps_decay_period"],
			600000,
			parameters["eval_eps"],
			self.writer,
			parameters["train_freq"]
		)
		policy1 = discrete_BCQ.discrete_BCQ(
			is_atari,
			self.num_action,
			self.state_dim,
			device,
			self.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
			self.writer,
			parameters['train_freq']
		)
		policy=[policy0,policy1]
		# if self.generate_buffer: policy.load(f"./models/{setting}")

		evaluations = []

		state, done = env.reset(), False
		episode_start = True
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0
		low_noise_ep = np.random.uniform(0,1) < self.low_noise_p
		best_eval = -1e9
		if inter == None:
			pass
		else:
			inter.policy = policy
		# Interact with the environment for max_timesteps
		if self.generate_buffer:
			inter.build_buffer()
		patience = 0
		for t in range(int(self.max_timestep)):

			episode_timesteps += 1

			# If generating the buffer, episode is low noise with p=low_noise_p.
			# If policy is low noise, we take random actions with p=eval_eps.
			# If the policy is high noise, we take random actions with p=rand_action_p.

				# if not low_noise_ep and np.random.uniform(0,1) < self.rand_action_p - parameters["eval_eps"]:
				# 	action = env.action_space.sample()
				# else:
				# 	action = policy.select_action(np.array(state), eval=True)

			if self.train_behavioral:
				if t < parameters["start_timesteps"]:
					action = env.action_space.sample()
				else:
					action = policy0.select_action(np.array(state))

				##### EDIT :  Indentation de ce qui suit pour le mettre dans train_behavioral, c'est déjà implémenté dans mon code (Tuple_ech)

				# Perform action and log results
				next_state, reward, done, info = env.step(action)
				episode_reward += reward
				norm_reward = reward/coeff_norm
				# Only consider "done" if episode terminates due to failure condition
				done_float = float(done) if episode_timesteps < env._max_episode_steps else 0
				#
				# # For atari, info[0] = clipped reward, info[1] = done_float
				# if is_atari:
				# 	reward = info[0]
				# 	done_float = info[1]

				# Store data in replay buffer
				replay_buffer.add(state, action, next_state, norm_reward, done_float, done, episode_start)
				state = copy.copy(next_state)
				episode_start = False

			# Train agent after collecting sufficient data
			if self.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
				# Q_loss = Q_loss + policy0.train(replay_buffer) #policy.train echantillonne dans le buffer un batch (64 par defaut) d'où l'intérêt de remplir le buffer avec de l'aléatoire avant de train
				policy0.train(replay_buffer)
				#### EDIT : Toujours une indentation car à la fin de l'appel de Tuple_ech, le buffer est rempli, on ne se soucie donc pas de ce qu'il se passe dans ni entre les episode de generate_buffer depuis le main
			if self.train_behavioral:
				if done:
					# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
					print(f'Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}')
					# Reset environment
					state, done = env.reset(), False
					episode_start = True
					episode_reward = 0
					episode_timesteps = 0
					episode_num += 1
					# low_noise_ep = np.random.uniform(0,1) < self.low_noise_p

			# Evaluate episode
			if self.train_behavioral and (t + 1) % parameters["eval_freq"] == 0 and t >= parameters["start_timesteps"]:
				patience += 1
				evaluations.append(self.eval_policy(policy0))
				np.save(f"./tests_optim/results/{setting}", evaluations)
				if evaluations[-1] > best_eval:
					patience = 0
					best_eval = evaluations[-1]
					policy0.save(f"./tests_optim/models/{setting}")
				if patience >= 1500 or best_eval == 0:
					np.save('deg_finale_batt', self.deg_finale_batt)
					print("end of training, patience threshold reached")
					break
				# if patience >= 500:# or best_eval == 0:
				# 	print("Chargement de la meilleure politique après avoir atteint le seuil de patience")
				# 	policy0.load(f"./tests_optim/models/MicrogridControlGym-v0_{self.manager._create_hashkey()}")
				# 	patience=0
				# else:
				# 	policy0.save(f"./models/actual_policy_{t}_{setting}")

		# Save final policy
		# if self.train_behavioral:
			# policy0.save(f"./models/{setting}")

# On enlève le else, on sauvegarde le replay buffer dans tous les cas pour voir ce qu'il s'y passe
		# else:
		replay_buffer.save(f"./tests_optim/buffers/{buffer_name}")
		np.save('deg_finale_batt', self.deg_finale_batt)
		# Save final buffer and performance
		if not self.generate_buffer:
			evaluations.append(self.eval_policy(policy0))
			np.save(f"./tests_optim/results/{setting}", evaluations)



	# Trains BCQ offline
	def train_BCQ(self, replay_buffer, is_atari, device):
		best_eval = -1e9
		# For saving files
		setting = f"{self.env}_{list(self.manager.dicto.keys())[-1]}"
		# setting = f"{self.env}_{self.seed}"
		buffer_name = f"{self.buffer_name}_{setting}"
		parameters = self.regular_parameters
		# Initialize and load policy
		policy = discrete_BCQ.discrete_BCQ(
			is_atari,
			self.num_action,
			self.state_dim,
			device,
			self.BCQ_threshold,
			parameters["discount"],
			parameters["optimizer"],
			parameters["optimizer_parameters"],
			parameters["polyak_target_update"],
			parameters["target_update_freq"],
			parameters["tau"],
			parameters["initial_eps"],
			parameters["end_eps"],
			parameters["eps_decay_period"],
			parameters["eval_eps"],
			self.writer,
			parameters['train_freq']
		)

		# Load replay buffer
		replay_buffer.load(f"./tests_optim/buffers/{buffer_name}")

		evaluations = []
		episode_num = 0
		done = True
		training_iters = 0
		patience = 0
		BCQ_eval_freq = 100
		while training_iters < self.max_timestep and patience < 15:
			# for _ in range(int(parameters["eval_freq"])):
			for _ in range(int(BCQ_eval_freq)):
				policy.train(replay_buffer)

			evaluations.append(self.eval_policy(policy))
			np.save(f"./tests_optim/results/BCQ_{setting}", evaluations)
			if evaluations[-1] > best_eval:# or best_eval == 0:
				patience = 0
				best_eval = evaluations[-1]
				policy.save(f"./tests_optim/models/{setting}")
			patience +=1
			training_iters += int(BCQ_eval_freq)
			print(f"Training iterations: {training_iters}")
			print("END OF A BCQ TRAINING LOOP")
		# policy.save(f"./models/{setting}")
		evaluations.append(self.eval_policy(policy))
		np.save(f"./tests_optim/results/BCQ_{setting}", evaluations)


	# Runs policy for X episodes and returns average reward
	# A fixed seed is used for the eval environment
	def eval_policy(self, policy, eval_episodes=1):
		print('MANAGER.DIM : ', self.manager)
		eval_env, _, _ = utils.make_env(self.env, manager=self.manager_test, battery = self.batt)
		action_list = np.array([])
		avg_reward = 0.
		rewards=np.array([])
		for _ in range(eval_episodes):
			state = eval_env.reset()
			done = False
			i=0
			while not done:
				i+=1
				action = policy.select_action(np.array(state), eval=True)
				action_list = np.append(action_list, action)
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward
				rewards=np.append(rewards, reward)

		avg_reward /= eval_episodes
		self.deg_finale_batt.append(self.batt.C)
		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("---------------------------------------")
		return avg_reward

	def get_score(self, scoretype=None, policy=None):
		years = [2005, 2006, 2007, 2008, 2011, 2020, 2010, 2009, 2012, 2016]
		dim_boundaries = {'PV': {'low': 0, 'high': 17}, 'batt': {'low': 0, 'high': 15}}
		import_data = Import_data(mode='train')
		consumption_norm, consumption, production_norm, production = import_data.split_years(5, 17,
																							 'PV_GIS_2005_2020_TOULOUSE_SARAH2.csv',
																							 periods=140256,
																							 start_time='2005-01-01',
																							 years=years)
		data = [consumption, consumption_norm, production, production_norm]
		manager = self.manager
		manager.data = data
		env, state_dim, num_action = utils.make_env("MicrogridControlGym-v0", manager, battery=self.batt)
		list_of_actions = np.array([])
		rewards = np.array([])
		obs = env.reset()
		states = []
		degra = []
		SOC = []
		is_done = False
		while not is_done:
			# action = env.action_space.sample()
			action = policy.select_action(np.array(obs), eval=True)
			states = np.append(states, obs)
			degra = np.append(degra, obs[-2])
			SOC = np.append(SOC, obs[0])
			obs, reward, is_done, info = env.step(action)
			rewards = np.append(rewards, reward)
			list_of_actions = np.append(list_of_actions, action)
		tab = env.render_to_file()
		grp = tab.groupby(tab.index.month)
		list_of_actions=np.append(list_of_actions, 3)
		tab['list_of_actions'] = list_of_actions
		tau_autocons = []
		tau_autoprod = []
		tab = env.render_to_file()
		grp = tab.groupby(tab.index.month)
		# action_charge = []
		# action_decharge = []
		# action_R = []
		prod = []
		# list_of_actions = np.append(list_of_actions, 3)
		for i in range(len(grp)):
			prod.append(sum(tab['PV production'][grp.groups[i + 1]]))
			tau_autoprod.append(
				1 - (sum(tab['Energy Bought'][grp.groups[i + 1]]) / sum(tab["Consumption"][grp.groups[i + 1]])))
			tau_autocons.append(
				1 - (sum(tab['Energy Sold'][grp.groups[i + 1]]) / sum(tab["PV production"][grp.groups[i + 1]])))
			# action_charge.append(len(np.where(tab['list_of_actions'][grp.groups[i + 1]] == 2)[0]))
			# action_decharge.append(len(np.where(tab['list_of_actions'][grp.groups[i + 1]] == 0)[0]))
			# action_R.append(len(np.where(tab['list_of_actions'][grp.groups[i + 1]] == 1)[0]))
		eval = np.sum(rewards)
		if scoretype == 'self-production':
			score = sum(tab['Energy Bought'])
		elif scoretype == 'both':
			score = sum(tab['Energy Bought']) + (sum(tab['Energy Sold']))
		elif scoretype == 'economic':
			score = sum(tab['Energy Bought']) - (sum(tab['Energy Sold']) * 0.25)
		return(score, tau_autoprod, tau_autocons, self.batt.C)

	def Iterate(self, temps, scoretype):
		# np.random.seed(seed=int(time.time()))

		len_episode = 8760
		env, self.state_dim, self.num_action = utils.make_env(self.env, self.manager, self.batt)
		parents = self.manager.choose_parents()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		tau_autoprod, tau_autocons = [], []
		if self.already_trained == 2:
			parameters = self.regular_parameters
			policy = discrete_BCQ.discrete_BCQ(
				False,
				self.num_action,
				self.state_dim,
				device,
				self.BCQ_threshold,
				parameters["discount"],
				parameters["optimizer"],
				parameters["optimizer_parameters"],
				parameters["polyak_target_update"],
				parameters["target_update_freq"],
				parameters["tau"],
				0,
				0,
				1,
				0,
				self.writer,
				parameters['train_freq']
			)
			policy.load(f"./tests_optim/models/MicrogridControlGym-v0_{self.manager._create_hashkey()}")
			score, tau_autoprod, tau_autocons, last_Cmax = self.get_score(scoretype=scoretype, policy=policy)
			print('score : ', score)
			return (score, 0, tau_autoprod, tau_autocons, last_Cmax)
		if self.already_trained == 1:
			parameters = self.regular_parameters
			policy = DQN.DQN(
				False,
				self.num_action,
				self.state_dim,
				device,
				parameters["discount"],
				parameters["optimizer"],
				parameters["optimizer_parameters"],
				parameters["polyak_target_update"],
				parameters["target_update_freq"],
				parameters["tau"],
				0,
				0,
				1,
				0,
				self.writer,
				parameters['train_freq']
			)
			policy.load(f"./tests_optim/models/MicrogridControlGym-v0_{self.manager._create_hashkey()}")
			score, tau_autoprod, tau_autocons, last_Cmax = self.get_score(scoretype=scoretype, policy=policy)
			print('score : ', score)
			return (score, 0, tau_autoprod, tau_autocons, last_Cmax)
		# Initialize buffer
		replay_buffer = utils.ReplayBuffer(self.state_dim, self.regular_parameters["batch_size"], self.regular_parameters["buffer_size"], device)
		# args = pd.DataFrame(
		# 	[[self.env, self.seed, self.manager, self.buffer_name, self.max_timestep, self.BCQ_threshold, self.low_noise_p,
		# 	 self.rand_action_p]],
		# 	columns=['env', 'seed', 'manager', 'buffer_name', 'max_timestep', 'BCQ_threshold', 'low_noise_p', 'rand_action_p'])
		# env = Monitor(env, self.manager.log_dir, allow_early_resets=False)
		# env = DummyVecEnv([lambda: env])

		if len(parents)<self.manager.nb_voisins:
			start = time.time()
			# if not enought close µgrid size, train from scratch in on-line setup
			self.train_behavioral = True
			self.generate_buffer = False
			self.cass = False
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			self.interact_with_environment(env, replay_buffer, False, device)
			end = time.time()
			temps["behavioral"] = np.append(temps["behavioral"],end-start)
			parameters = self.regular_parameters
			policy = DQN.DQN(
				False,
				self.num_action,
				self.state_dim,
				device,
				parameters["discount"],
				parameters["optimizer"],
				parameters["optimizer_parameters"],
				parameters["polyak_target_update"],
				parameters["target_update_freq"],
				parameters["tau"],
				0,
				0,
				1,
				0,
				self.writer,
				parameters['train_freq']
			)
			policy.load(f"./tests_optim/models/MicrogridControlGym-v0_{self.manager._create_hashkey()}")
			score, tau_autoprod, tau_autocons, last_Cmax = self.get_score(scoretype=scoretype, policy=policy)
			print('score : ', score)
		else:
			# else, use buffers to train off-line
			start = time.time()
			self.train_behavioral = False
			self.generate_buffer = True
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			#replay_buffer = utils.ReplayBuffer(self.state_dim, self.regular_parameters["batch_size"], self.regular_parameters["buffer_size"], device)
			self.INTER = Interact(self.manager, log=1, buffer_size=self.regular_parameters["buffer_size"], replay_buffer=replay_buffer, low_noise_p=self.low_noise_p, rand_action_p=self.rand_action_p, battery=self.batt)
			print("INTERACT TO GENERATE BUFFER NOW")
			self.interact_with_environment(env, replay_buffer, False, device, inter=self.INTER)
			print("TRAIN BCQ NOW")
			self.train_BCQ(replay_buffer, False, device)
			end = time.time()
			temps["bcq"] = np.append(temps["bcq"],end - start)
			parameters = self.regular_parameters
			policy = discrete_BCQ.discrete_BCQ(
				False,
				self.num_action,
				self.state_dim,
				device,
				self.BCQ_threshold,
				parameters["discount"],
				parameters["optimizer"],
				parameters["optimizer_parameters"],
				parameters["polyak_target_update"],
				parameters["target_update_freq"],
				parameters["tau"],
				0,
				0,
				1,
				0,
				self.writer,
				parameters['train_freq']
			)
			policy.load(f"./tests_optim/models/MicrogridControlGym-v0_{self.manager._create_hashkey()}")
			score, tau_autoprod, tau_autocons, last_Cmax = self.get_score(scoretype=scoretype, policy=policy)
			print('score : ', score)
		return(score, temps, tau_autoprod, tau_autocons, last_Cmax)
	# # Set seeds
	# env.seed(self.seed)
	# env.action_space.seed(self.seed)
	# torch.manual_seed(self.seed)
	# np.random.seed(self.seed)
