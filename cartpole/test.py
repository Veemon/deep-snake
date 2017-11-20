from model import *

import sys
import time
import copy

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from train import live_plot, state_tensor

def test():
    # initialize agent
    agent = Agent(num_episodes=10000)

    # cli
    ver = 0
    if len(sys.argv) > 1:
        try:
            ver = int(sys.argv[1])
        except:
            pass

    # load previous checkpoint
    epoch_offset = 0
    if checkpoint_available("saves") == True:
        load_checkpoint("saves", agent, version=ver)

    # initialize environment
    env = gym.make('CartPole-v0').unwrapped

    # test
    episode_durations = []
    while 1:

        # Initialize the environment and state
        env.reset()
        last_screen = state_tensor(env)
        current_screen = state_tensor(env)
        s = current_screen - last_screen

        for t in count():
            # Select and perform an action
            a = agent.select_action(s)
            _, _, done, _ = env.step(a)

            # Observe new state
            last_screen = current_screen
            current_screen = state_tensor(env)

            # Plot
            if done:
                episode_durations.append(t + 1)
                live_plot(episode_durations, 1, 'episode duration')
                break

if __name__ == '__main__':
    try:
        plt.ion()
        test()
    except KeyboardInterrupt:
        pass
