import argparse
import copy
import importlib
import json
import os

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
class main_BCQ():
	def __init__(self, env, manager, seed=0, buffer_name="Default", max_timestep=1e6, BCQ_threshold=0.3, low_noise_p=0.01, rand_action_p=0.3):


		self.regular_parameters = {
			# multi_env
			"nb_parents": 2,
			"euclidian_dist": 4,
			# Exploration
			### Modifié pour l'abaisser dans les épisodes en low noise
			"start_timesteps": 1e3, #nombre de step avant de ne plus prendre d'action aléatoires
			"initial_eps": 0.03,
			"end_eps": 0.03,
			"eps_decay_period": 1,
			# Evaluation
			"eval_freq": 8759,#Attention c'est en nombre de step et pas en nombre d'épisodes
			"eval_eps": 0,
			# Learning
			"discount": 0.99,
			"buffer_size": 1e6,
			"batch_size": 64,
			"optimizer": "Adam",
			"optimizer_parameters": {
				"lr": 3e-4
			},
			"train_freq": 1,
			"polyak_target_update": True,
			"target_update_freq": 1,
			"tau": 0.005
		}

		# Load parameters
		self.env = env
		self.seed = seed
		self.buffer_name = buffer_name
		self.max_timestep = max_timestep
		self.BCQ_threshold = BCQ_threshold
		self.low_noise_p = low_noise_p
		self.rand_action_p = rand_action_p
		self.manager = manager



	if not os.path.exists("./results"):
		os.makedirs("./results")

	if not os.path.exists("./models"):
		os.makedirs("./models")

	if not os.path.exists("./buffers"):
		os.makedirs("./buffers")

	def interact_with_environment(self, env, replay_buffer, is_atari, device, inter=None):
		# For saving files
		manager = self.manager
		setting = f"{self.env}_{list(manager.dicto.keys())[-1]}" #On met le hashkey correspondant au dimensionnement actuel dans le nom. On retrouve ce dimensionnement comme le dernier ajouté au dictionnaire du manager
		buffer_name = f"{self.buffer_name}_{setting}"
		print("SETING : ", setting)
		print("MANAGER : ", manager)
		# Initialize and load policy
		parameters=self.regular_parameters
		policy = DQN.DQN(
			is_atari,
			self.num_actions,
			self.state_dim,
			device,
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
		)

		if self.generate_buffer: policy.load(f"./models/{setting}")

		evaluations = []

		state, done = env.reset(), False
		episode_start = True
		episode_reward = 0
		episode_timesteps = 0
		episode_num = 0
		low_noise_ep = np.random.uniform(0,1) < self.low_noise_p
		best_eval = 0
		if inter == None:
			pass
		else:
			inter.policy = policy
		# Interact with the environment for max_timesteps
		for t in range(int(self.max_timestep)):

			episode_timesteps += 1

			# If generating the buffer, episode is low noise with p=low_noise_p.
			# If policy is low noise, we take random actions with p=eval_eps.
			# If the policy is high noise, we take random actions with p=rand_action_p.
			if self.generate_buffer:
				inter.build_buffer()
				# if not low_noise_ep and np.random.uniform(0,1) < self.rand_action_p - parameters["eval_eps"]:
				# 	action = env.action_space.sample()
				# else:
				# 	action = policy.select_action(np.array(state), eval=True)

			if self.train_behavioral:
				if t < parameters["start_timesteps"]:
					action = env.action_space.sample()
				else:
					action = policy.select_action(np.array(state))

				##### EDIT :  Indentation de ce qui suit pour le mettre dans train_behavioral, c'est déjà implémenté dans mon code (Tuple_ech)

				# Perform action and log results
				next_state, reward, done, info = env.step(action)
				episode_reward += reward

				# Only consider "done" if episode terminates due to failure condition
				done_float = float(done) if episode_timesteps < env._max_episode_steps else 0
				#
				# # For atari, info[0] = clipped reward, info[1] = done_float
				# if is_atari:
				# 	reward = info[0]
				# 	done_float = info[1]

				# Store data in replay buffer
				replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
				state = copy.copy(next_state)
				episode_start = False

			# Train agent after collecting sufficient data
			if self.train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
				policy.train(replay_buffer) #policy.train echantillonne dans le buffer un batch (64 par defaut) d'où l'intérêt de remplir le buffer avec de l'aléatoire avant de train

				#### EDIT : Toujours une indentation car à la fin de l'appel de Tuple_ech, le buffer et rempli, on ne se soucie donc pas de ce qu'il se passe dans ni entre les episode de generate_buffer depuis le main

				if done:
					# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
					print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
					# Reset environment
					state, done = env.reset(), False
					episode_start = True
					episode_reward = 0
					episode_timesteps = 0
					episode_num += 1
					low_noise_ep = np.random.uniform(0,1) < self.low_noise_p

			# Evaluate episode
			if self.train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
				evaluations.append(self.eval_policy(policy))
				np.save(f"./results/{setting}", evaluations)
				if evaluations[-1]>best_eval or best_eval == 0:
					best_eval = evaluations[-1]
					policy.save(f"./models/{setting}")

		# Save final policy
		if self.train_behavioral:
			policy.save(f"./models/{setting}")

		# Save final buffer and performance
		else:
			evaluations.append(self.eval_policy(policy))
			np.save(f"./results/{setting}", evaluations)
			replay_buffer.save(f"./buffers/{buffer_name}")


	# Trains BCQ offline
	def train_BCQ(self, replay_buffer, is_atari, device):
		# For saving files
		setting = f"{self.env}_{list(self.manager.dicto.keys())[-1]}"
		# setting = f"{self.env}_{self.seed}"
		buffer_name = f"{self.buffer_name}_{setting}"
		parameters = self.regular_parameters
		# Initialize and load policy
		policy = discrete_BCQ.discrete_BCQ(
			is_atari,
			self.num_actions,
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
			parameters["eval_eps"]
		)

		# Load replay buffer
		replay_buffer.load(f"./buffers/{buffer_name}")

		evaluations = []
		episode_num = 0
		done = True
		training_iters = 0

		while training_iters < self.max_timestep:

			for _ in range(int(parameters["eval_freq"])):
				policy.train(replay_buffer)

			evaluations.append(self.eval_policy(policy))
			np.save(f"./results/BCQ_{setting}", evaluations)

			training_iters += int(parameters["eval_freq"])
			print(f"Training iterations: {training_iters}")


	# Runs policy for X episodes and returns average reward
	# A fixed seed is used for the eval environment
	def eval_policy(self, policy, eval_episodes=5):
		print('MANAGER.DIM : ', self.manager)
		eval_env, _, _ = utils.make_env(self.env, manager=self.manager)
		eval_env.seed(self.seed + 100)

		avg_reward = 0.
		for _ in range(eval_episodes):
			state, done = eval_env.reset(), False
			while not done:
				action = policy.select_action(np.array(state), eval=True)
				state, reward, done, _ = eval_env.step(action)
				avg_reward += reward

		avg_reward /= eval_episodes

		print("---------------------------------------")
		print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
		print("---------------------------------------")
		return avg_reward




	def Iterate(self):
		len_episode = 8760
		env, self.state_dim, self.num_actions = utils.make_env(self.env, self.manager)
		parents = self.manager.choose_parents()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Initialize buffer
		replay_buffer = utils.ReplayBuffer(self.state_dim, self.regular_parameters["batch_size"], self.regular_parameters["buffer_size"], device)
		# args = pd.DataFrame(
		# 	[[self.env, self.seed, self.manager, self.buffer_name, self.max_timestep, self.BCQ_threshold, self.low_noise_p,
		# 	 self.rand_action_p]],
		# 	columns=['env', 'seed', 'manager', 'buffer_name', 'max_timestep', 'BCQ_threshold', 'low_noise_p', 'rand_action_p'])
		# env = Monitor(env, self.manager.log_dir, allow_early_resets=False)
		# env = DummyVecEnv([lambda: env])

		if len(parents)<self.manager.nb_voisins:
			# if not enought close µgrid size, train from scratch in on-line setup
			self.train_behavioral = True
			self.generate_buffer = False
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			self.interact_with_environment(env, replay_buffer, False, device)
		else:
			# else, use buffers to train off-line
			self.train_behavioral = False
			self.generate_buffer = True
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			replay_buffer = utils.ReplayBuffer(self.state_dim, self.regular_parameters["batch_size"], self.regular_parameters["buffer_size"], device)
			self.INTER = Interact(self.manager, log=1, buffer_size=self.regular_parameters["buffer_size"], replay_buffer=replay_buffer, low_noise_p=self.low_noise_p, rand_action_p=self.rand_action_p)
			self.interact_with_environment(env, replay_buffer, False, device, inter=self.INTER)
			self.train_BCQ(env, replay_buffer, False, device)

	# # Set seeds
	# env.seed(self.seed)
	# env.action_space.seed(self.seed)
	# torch.manual_seed(self.seed)
	# np.random.seed(self.seed)
