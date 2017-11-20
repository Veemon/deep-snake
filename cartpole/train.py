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


def state_tensor(env):
    # RGB Array -> Channels, Height, Width
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))

    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    screen_width = 600

    # Get cart location
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    cart_location = int(env.state[0] * scale + screen_width / 2.0)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)

    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    resize = T.Compose([T.ToPILImage(),
                        T.Scale(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    return resize(screen).unsqueeze(0).type(torch.FloatTensor)

def live_plot(x, fig, title):
    plt.figure(fig)
    plt.clf()
    plt.title(title)
    plt.plot(x)
    plt.pause(0.001)

def train():
    # training parameters
    checkpoint = 10

    batch_size = 32
    num_epochs = 100
    decay_epoch = 50

    net_switch = 10

    gamma = 0.95
    lr = 0.00025

    # counters
    loss_values = []
    episode_durations = []

    # initialize agent
    agent = Agent(capacity=10000,
                    batch_size=batch_size,
                    num_episodes=decay_epoch,
                    gamma=gamma,
                    lr=lr,
                    net_switch=net_switch)

    # load previous checkpoint
    epoch_offset = 0
    if checkpoint_available("saves") == True:
        epoch_offset = load_checkpoint("saves", agent)

    # initialize environment
    env = gym.make('CartPole-v0').unwrapped

    for epoch in range(num_epochs + 1):

        # Initialize the environment and state
        env.reset()
        last_screen = state_tensor(env)
        current_screen = state_tensor(env)
        s = current_screen - last_screen

        for t in count():
            # Select and perform an action
            a = agent.select_action(s)
            _, r, done, _ = env.step(a)

            r = 0.0 if done == True else 1.0
            r = torch.FloatTensor([r])

            # Observe new state
            last_screen = current_screen
            current_screen = state_tensor(env)
            if not done:
                s1 = current_screen - last_screen
            else:
                s1 = None

            # Store the transition in memory
            a = torch.LongTensor([[a]])
            agent.push(s, a, s1, r)

            # Move to the next state
            s = s1

            # Perform one step of the optimization (on the target network)
            loss_values.append(agent.optimize())
            if done:
                episode_durations.append(t + 1)
                live_plot(loss_values, 1, 'loss')
                live_plot(episode_durations, 2, 'episode duration')
                break

        # save model
        if epoch != 0 and epoch % checkpoint == 0:
            save_checkpoint(agent, 'saves/epoch_{}.model'.format(epoch_offset + epoch))

if __name__ == '__main__':
    try:
        plt.ion()
        train()
        input()
    except KeyboardInterrupt:
        pass
