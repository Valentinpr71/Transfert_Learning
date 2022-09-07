import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import pandas as pd
import gym

from Callbacks.SavingBestRewards import SaveOnBestTrainingRewardCallback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from . import discrete_BCQ
from . import DQN
from . import utils
from Tuple_ech import Interact


def interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters, generate_buffer, train_behavioral):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = DQN.DQN(
		is_atari,
		num_actions,
		state_dim,
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

	if generate_buffer: policy.load(f"./models/behavioral_{setting}")
	
	evaluations = []

	state, done = env.reset(), False
	episode_start = True
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	low_noise_ep = np.random.uniform(0,1) < args.low_noise_p
	best_eval = 0
	# Interact with the environment for max_timesteps
	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# If generating the buffer, episode is low noise with p=low_noise_p.
		# If policy is low noise, we take random actions with p=eval_eps.
		# If the policy is high noise, we take random actions with p=rand_action_p.
		if generate_buffer:
			if not low_noise_ep and np.random.uniform(0,1) < args.rand_action_p - parameters["eval_eps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state), eval=True)

		if train_behavioral:
			if t < parameters["start_timesteps"]:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state))

		# Perform action and log results
		next_state, reward, done, info = env.step(action)
		episode_reward += reward

		# Only consider "done" if episode terminates due to failure condition
		done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

		# For atari, info[0] = clipped reward, info[1] = done_float
		if is_atari:
			reward = info[0]
			done_float = info[1]
			
		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_float, done, episode_start)
		state = copy.copy(next_state)
		episode_start = False

		# Train agent after collecting sufficient data
		if train_behavioral and t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
			policy.train(replay_buffer)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_start = True
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
			low_noise_ep = np.random.uniform(0,1) < args.low_noise_p

		# Evaluate episode
		if train_behavioral and (t + 1) % parameters["eval_freq"] == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/behavioral_{setting}", evaluations)
			if evaluations[-1]>best_eval or best_eval == 0:
				best_eval=evaluations[-1]
				policy.save(f"./models/behavioral_{setting}")

	# Save final policy
	if args.train_behavioral:
		policy.save(f"./models/behavioral_{setting}")

	# Save final buffer and performance
	else:
		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/buffer_performance_{setting}", evaluations)
		replay_buffer.save(f"./buffers/{buffer_name}")


# Trains BCQ offline
def train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize and load policy
	policy = discrete_BCQ.discrete_BCQ(
		is_atari,
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
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
	
	while training_iters < args.max_timesteps: 
		
		for _ in range(int(parameters["eval_freq"])):
			policy.train(replay_buffer)

		evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _ = utils.make_env(env_name)
	eval_env.seed(seed + 100)

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


class main_BCQ():
	def __init__(self, env, manager, seed=0, buffer_name="Default", max_timestep=1e6, BCQ_threshold=0.3, low_noise_p=0.01, rand_action_p=0.3):


		self.regular_parameters = {
			# multi_env
			"nb_parents": 2,
			"euclidian_dist": 4,
			# Exploration
			"start_timesteps": 1e3,
			"initial_eps": 0.1,
			"end_eps": 0.1,
			"eps_decay_period": 1,
			# Evaluation
			"eval_freq": 5e3,
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



	# if not os.path.exists("./results"):
	# 	os.makedirs("./results")
	#
	# if not os.path.exists("./models"):
	# 	os.makedirs("./models")
	#
	# if not os.path.exists("./buffers"):
	# 	os.makedirs("./buffers")

	def Iterate(self):
		len_episode = 8760
		env, state_dim, num_actions = utils.make_env(self.env, self.manager)
		parameters =  self.regular_parameters
		parents = self.manager.choose_parents()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Initialize buffer
		replay_buffer = utils.ReplayBuffer(state_dim, parameters["batch_size"], parameters["buffer_size"], device)
		args = pd.DataFrame(
			[[self.env, self.seed, self.buffer_name, self.max_timestep, self.BCQ_threshold, self.low_noise_p,
			 self.rand_action_p]],
			columns=['env', 'seed', 'buffer_name', 'max_timestep', 'BCQ_threshold', 'low_noise_p', 'rand_action_p'])
		# env = Monitor(env, self.manager.log_dir, allow_early_resets=False)
		# env = DummyVecEnv([lambda: env])

		if len(parents)<self.manager.nb_voisins:
			# if not enought close µgrid size, train from scratch in on-line setup
			self.train_behavioral = True
			self.generate_buffer = False
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			interact_with_environment(env, replay_buffer, False, num_actions, state_dim, device, args, parameters, self.generate_buffer, self.train_behavioral)
		else:
			# else, use buffers to train off-line
			self.train_behavioral = False
			self.generate_buffer = True
			print("generate_buffer",self.generate_buffer,"train_behavioral", self.train_behavioral)
			interact_with_environment(env, replay_buffer, False, num_actions, state_dim, device, args, parameters, self.generate_buffer, self.train_behavioral)
			train_BCQ(env, replay_buffer, False, num_actions, state_dim, device, args, parameters)

	# # Set seeds
	# env.seed(self.seed)
	# env.action_space.seed(args.seed)
	# torch.manual_seed(args.seed)
	# np.random.seed(args.seed)
