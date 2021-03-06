import os
import sys
import copy
import operator
import random
import math
import time

from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(int(time.time()))
torch.cuda.manual_seed_all(int(time.time()))

import better_exceptions
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def length(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Value
        self.v1 = nn.Linear(64, 32)
        self.v2 = nn.Linear(32, 1)

        # Advantage
        self.a1 = nn.Linear(64, 32)
        self.a2 = nn.Linear(32, 4)


    def forward(self, x):
        # convolution
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))

        # split
        x = x.view(x.size(0), -1)
        v, a = torch.split(x, 64, 1)

        # value
        v = nn.functional.relu(self.v1(v))
        v = self.v2(v)

        # advantage
        a = nn.functional.relu(self.a1(a))
        a = self.a2(a)

        # q-function
        return v + a

class Agent:
    def __init__(self, capacity=0, batch_size=0, gamma=0.999,
                init_lr=0.1, final_lr=0.1, lr_decay=0, epsilon_decay=0,
                net_switch=2, final_epsilon=0.1, fixed_epsilon=False):

        # members
        self.capacity = capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.lr_decay = lr_decay
        self.epsilon_decay = epsilon_decay
        self.net_switch = net_switch
        self.final_epsilon = final_epsilon
        self.fixed_epsilon = fixed_epsilon

        # objects
        self.memory = Memory(capacity)

        self.primary = DQN().cuda()
        self.target = DQN().cuda()

        self.optimizer = optim.Adam(self.primary.parameters(), lr=init_lr)

        # inner
        self.init_epsilon = 0.9
        self.num_epochs = 0

        self.lr = init_lr
        self.epsilon = self.init_epsilon

    def select_action(self,state, debug=False):
        # epsilon annealing
        if self.fixed_epsilon == False:
            self.epsilon = self.final_epsilon + (self.init_epsilon - self.final_epsilon)
            self.epsilon *= math.exp(-1. * self.num_epochs / self.epsilon_decay)
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon
        else:
            self.epsilon = self.final_epsilon

        # choice
        if random.random() > self.epsilon:
            state = Variable(state, volatile=True).cuda()
            q_vals = self.primary(state).cpu()
            choice = q_vals.data.max(1)[1].view(1, 1)
            q_vals = torch.t(q_vals)
        else:
            q_vals = [None, None, None, None]
            choice = torch.LongTensor([[random.randrange(4)]])

        if debug == False:
            return int(choice.numpy()[0][0])
        else:
            return int(choice.numpy()[0][0]), q_vals

    def optimize(self):

        # leave if too early
        if self.memory.length() < self.batch_size:
            return
        else:
            self.num_epochs += 1

        # learning rate decay
        self.lr = self.final_lr + (self.init_lr - self.final_lr)
        self.lr *= math.exp(-1. * self.num_epochs / self.lr_decay)
        if self.lr < self.final_lr:
            self.lr = self.final_lr

        # update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        # get batch sample
        samples = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*samples))

        # mask terminal states
        terminal_mask = torch.cuda.ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        # unpack
        s1 = Variable(torch.cat([s for s in batch.next_state
                                        if s is not None]),
                                         volatile=True).cuda()

        s = Variable(torch.cat(batch.state)).cuda()
        a = Variable(torch.cat(batch.action)).cuda()
        r = Variable(torch.cat(batch.reward)).cuda()

        # computate Q(s,a) via primary network
        q = self.primary(s).gather(1, a)

        # compute Q(s',a') via target network
        q1 = Variable(torch.zeros(self.batch_size).type(torch.cuda.FloatTensor).add(-1))
        q1[terminal_mask] = self.target(s1).max(1)[0]
        q1.volatile = False

        # discounted reward
        q1[terminal_mask] = (q1[terminal_mask] * self.gamma) + r[terminal_mask]
        q1 = torch.clamp(q1, min=-1.0, max=1.0)

        # compute loss
        loss = nn.functional.smooth_l1_loss(q, q1)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.primary.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update target network
        if self.num_epochs % self.net_switch == 0:
            self.target = copy.deepcopy(self.primary)

        return loss.cpu().data.numpy()[0]

def checkpoint_available(path):
    files = list( filter( lambda f: f[- (len('.model')):] == '.model', next(os.walk(path))[2] ) )
    return True if len(files) > 0 else False

def load_checkpoint(path, agent, version=0):
    # get all versions
    files = list( filter( lambda f: f[- (len('.model')):] == '.model', next(os.walk(path))[2] ) )
    numbers = []
    for f in files:
        number = f.split('_')[1].split('.')[0]
        numbers.append(int(number))

    save = -1

    # find max
    if version == 0:
        save = files[np.argmax(numbers)]
        selected = max(numbers)

    # find specific version
    else:
        for i in range(len(numbers)):
            if numbers[i] == version:
                save = files[i]
                selected = numbers[i]

    # safey
    if save == -1:
        print('could not find save.')
        sys.exit()

    # load
    agent.primary.load_state_dict(torch.load(path + '/' + save))
    agent.target = copy.deepcopy(agent.primary)
    return selected

def save_checkpoint(agent, path):
    torch.save(agent.primary.state_dict(), path)
