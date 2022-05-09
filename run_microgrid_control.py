import numpy as np
import os
import matplotlib.pyplot as plt
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import logging

from default_parser import process_args
from agent import NeuralAgent
from learning_algos.q_net_keras import MyQNetwork
from envs.microgrid_control_env import microgrid_control



class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 365 * 24 - 1
    EPOCHS = 10
    STEPS_PER_TEST = 365 * 24 - 1
    PERIOD_BTW_SUMMARY_PERFS = -1  # Set to -1 for avoiding call to env.summarizePerformance

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0002
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.98
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .3
    EPSILON_DECAY = 500000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    if (parameters.param1 is not None and parameters.param1 != "1"):
        # We Reduce the size of the time series so that the number of days is divisible by 4*parameters.param1
        # That way, the number of days in each season is divisible by parameters.param1 and it is thus possible
        # to reduce the variety of the data within each season in the time series by a factor of parameters.param1
        parameters.steps_per_epoch = parameters.steps_per_epoch - (
                    parameters.steps_per_epoch % (24 * 4 * int(parameters.param1))) - 1

    # --- Instantiate environment ---
    env = microgrid_control(rng)#, parameters.param1, parameters.param2, parameters.param3)

    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))), #Là il va y avoir un problème de shape
        parameters.batch_size,
        rng)
    agent.run(parameters.epochs, parameters.steps_per_epoch)