#!/usr/bin/env python3

import os
import sys
import time
import operator
import random
import math
import pygame

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

import numpy as np

from snake import *
from model import *

def distance_reward(snake_pos, fruit_pos):
    head = snake_pos[0]
    hx = head[0]
    hy = head[1]

    fx = fruit_pos[0]
    fy = fruit_pos[1]

    x = (fx - hx)
    x = x*x

    y = (fy - hy)
    y = y*y

    dist = math.sqrt(x + y)
    norm = 1 - ((dist*dist) / (map_size * map_size) * 2)

    if norm > 0.975:
        norm = 0.975
    if norm < 0.0:
        norm = 0.0

    return norm

def debug_plot(state):
    plt.figure(1)
    plt.imshow(state.squeeze(0).permute(2, 1, 0).numpy(),
               interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
    input()

def state_tensor(surface):
    # RGB Array -> Channels, Height, Width
    pixels = pygame.surfarray.array3d(surface).transpose((2, 0, 1))

    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    pixels = np.ascontiguousarray(pixels, dtype=np.float32) / 255
    pixels = torch.from_numpy(pixels)

    # Resize, and add a batch dimension (BCHW)
    resize = T.Compose([T.ToPILImage(),
                        T.Scale(40, interpolation=Image.CUBIC),
                        T.ToTensor()])

    return resize(pixels).unsqueeze(0).type(torch.FloatTensor)

def live_plot(in_x, fig, title):

    # little performance boost
    x = in_x[-100:]
    if len(x) < 2:
        return

    # grab figure handle
    plt.figure(fig)
    plt.clf()

    # update
    plt.title(title)
    plt.plot(x)
    plt.pause(0.001)

# counters
loss_values = []
epsiode_rewards = []
epsilon_values = []

def train():
    # training parameters
    checkpoint = 1000

    batch_size = 128
    num_epochs = 200000
    decay_epoch = 400000

    net_switch = 30

    gamma = 0.65
    lr = 0.00025

    final_epsilon = 0.05

    # counters
    global loss_values
    global epsiode_rewards
    global epsilon_values

    # initialize agent
    agent = Agent(capacity=4096,
                    batch_size=batch_size,
                    num_episodes=decay_epoch,
                    gamma=gamma,
                    lr=lr,
                    final_epsilon=final_epsilon,
                    net_switch=net_switch)

    # CLI
    exploit = False
    for arg in sys.argv:
        if arg == '--exploit':
            exploit = True

    # load previous checkpoint
    epoch_offset = 0
    if checkpoint_available("saves") == True:
        num = epoch_offset = load_checkpoint("saves", agent)
        print('loaded version {} ... '.format(num), end='')
        if exploit == False:
            print()
            agent.init_epsilon = 0.75
        else:
            print('exploit mode')
            agent.init_epsilon = 0.3

    # Game Constants
    global map_size
    init_pos = map_size/2
    if init_pos != int(init_pos):
        map_size -= 1
        init_pos = map_size/2

    for epoch in range(num_epochs + 1):

        # Game
        game_map = []
        initialise_map(game_map, map_size)
        fill_val = start_fill

        # Systems
        sqr_size = map_size*pixel_size
        screen = pygame.display.set_mode((sqr_size,sqr_size))
        screen.fill((fill_val,fill_val,fill_val))

        # Game Arg Unpacking
        game_args = reset_states(game_map, map_size)

        # Systems
        tick_rate = game_args.tick_rate
        move_rate = game_args.move_rate

        last_t = game_args.last_t
        last_move = game_args.last_move
        last_length = game_args.last_length

        frame_times = game_args.frame_times[:]
        t = game_args.t

        # Game
        direction = game_args.direction
        velocity = game_args.velocity
        length = game_args.length

        snake_pos = game_args.snake_pos[:]
        fruit_pos = game_args.fruit_pos[:]

        # Initial Render
        draw_map(screen, game_map, map_size, fill_val + 2)
        draw_fruit(screen, fruit_pos, velocity, t, ai=True)
        draw_snake(screen, snake_pos, length, velocity, t, ai=True)

        pygame.display.flip()

        # Grab state and set max reward
        s = state_tensor(screen)
        max_reward = 0

        for t in count():

            # get momentum
            body_direction = get_body_direction(game_args.snake_pos)

            # get action
            desired = agent.select_action(s)
            if desired == up and body_direction != down:
                direction = up
            if desired == right and body_direction != left:
                direction = right
            if desired == down and body_direction != up:
                direction = down
            if desired == left and body_direction != right:
                direction = left

            # update
            move_snake(snake_pos, velocity, direction, length)
            velocity = collide_self(snake_pos, length, velocity)
            velocity = collide_wall(game_map, map_size, snake_pos, velocity)
            fruit_pos, length = collide_fruit(snake_pos, length, fruit_pos, game_map, map_size)

            # draw
            fill_val = start_fill
            screen.fill((fill_val,fill_val,fill_val))

            draw_map(screen, game_map, map_size, fill_val + 2)
            draw_fruit(screen, fruit_pos, velocity, t, ai=True)
            draw_snake(screen, snake_pos, length, velocity, t, ai=True)

            pygame.display.flip()

            # initial reward
            reward = 0

            # if growth
            if length - last_length > 0:
                reward = 1
                max_reward += 1
                last_length = length

            # observe new state
            if velocity != 0:
                s1 = state_tensor(screen)
            else:
                reward = -1
                s1 = None

            # memorize
            a = torch.LongTensor([[desired]])
            r = torch.FloatTensor([reward])
            agent.memory.push(s, a, s1, r)

            # update
            s = s1

            # optimize
            l = agent.optimize()
            if l is not None:
                loss_values.append(l)
            if velocity == 0:
                epsiode_rewards.append(max_reward)
                epsilon_values.append(agent.epsilon)
                live_plot(loss_values, 1, 'loss')
                live_plot(epsiode_rewards, 2, 'episode max rewards')
                live_plot(epsilon_values, 3, 'epsilon')
                break

        # save model
        if epoch != 0 and epoch % checkpoint == 0:
            save_checkpoint(agent, 'saves/epoch_{}.model'.format(epoch_offset + epoch))

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Snek', 'None')

    plt.ion()

    train()

    plt.plot(loss_values)
    plt.plot(epsiode_rewards)
    plt.plot(epsilon_values)
    plt.show()

    input()
