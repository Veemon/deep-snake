import argparse
import time
import copy
import sys
import os 

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from colorama import Style, init
init()

from environment import SnakeEnvironment
from environment import REWARD_LOSS, REWARD_DEFAULT, REWARD_POINT 
from environment import MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT
from environment import MAP_EMPTY, MAP_WALL, MAP_SNAKE_BODY, MAP_SNAKE_HEAD, MAP_APPLE

# thanks to @tanyaschlusser, i really couldn't be bothered to read the gif spec
# one thing I do want to note, it is EXTREMELY memory inefficient
# -> https://github.com/tanyaschlusser/array2gif
from gif_writer import write_gif

LEARN_RATE          = 5e-4
STAGNATION_DURATION = 1e5
GAME_SIZE_RANDOM    = True
GAME_SIZE_LOWER     = 8
GAME_SIZE_UPPER     = 11
GAME_SIZE           = 11

LOGGING_FREQ     = 500
CHECKPOINT_FREQ  = 5e4
TARGET_STEPS     = 1e7
REPLAY_GAMES     = 9
REPLAY_VERBOSITY = False

ACTION_SIZE = 4
EPSILON     = 1e-4

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.ref = nn.ReflectionPad2d(4) 
        self.c1  = nn.Conv2d(1,  64, 2, 1, padding=0, dilation=1)
        self.c2  = nn.Conv2d(64, 64, 2, 1, padding=0, dilation=2)
        self.c3  = nn.Conv2d(64,  4, 2, 1, padding=0, dilation=1)

    def forward(self, x):
        x = F.elu(self.c1(self.ref(x)))
        x = F.elu(self.c2(self.ref(x)))
        x = F.elu(self.c3(x))
        x = self.avg(x).view(-1)
        return x

def stagnation(x, w, h):
    a = (h - 2) * (w - 2)
    b = (w + h)
    m = (a - b) / STAGNATION_DURATION
    return min( int(m*x + b), a)

def train(agent, path, last_step):
    # setup
    if GAME_SIZE_RANDOM:
        game_width  = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
        game_height = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
    else:
        game_width  = GAME_SIZE
        game_height = GAME_SIZE

    game      = SnakeEnvironment(game_height, game_width)
    optimizer = optim.Adam(agent.parameters(), lr=LEARN_RATE)
    kl_loss   = nn.KLDivLoss(reduction='batchmean')

    steps_since_reward = 0
    reward_acc         = 0
    loss_acc           = 0
    total_step         = 0
    log_step           = last_step

    # get initial state
    s = game.env()
    s = torch.FloatTensor(s).view(1,1,game_height,game_width)

    # setup target dictionaries
    target_loss_temp = torch.zeros(ACTION_SIZE) + (1 / (ACTION_SIZE-1)) - EPSILON
    target_loss = {}
    for i in range(ACTION_SIZE):
        target_loss[i] = target_loss_temp.clone()
        target_loss[i][i] = 0

    target_point_temp = torch.zeros(ACTION_SIZE)
    target_point = {}
    for i in range(ACTION_SIZE):
        target_point[i] = target_point_temp.clone()
        target_point[i][i] = 1 - EPSILON

    # keep history for backprop
    history = []
    t1 = time.time()
    while log_step < TARGET_STEPS + last_step:
        # sample environment
        a    = agent(s)
        amax = a.argmax().item()
        r, s = game.step(amax)
        s    = torch.FloatTensor(s).view(1,1,game_height,game_width)
        
        # store both of these to avoid double calculations
        history.append([a,amax])

        # record this for logging
        total_step += 1
        reward_acc += r

        # apply stagnation countermeasure
        if r == REWARD_DEFAULT:
            steps_since_reward += 1
            if steps_since_reward > stagnation(log_step, game_height, game_width):
                r = REWARD_LOSS
        
        # reinforce if we get a signal
        if r != REWARD_DEFAULT:
            # we got some sort of signal, cant stagnate with that
            steps_since_reward = 0

            # make a batch of target distributions
            action_batch = [F.log_softmax(x[0], dim=-1) for x in history]
            if r == REWARD_LOSS:
                target_batch = [target_loss[x[1]] for x in history]
            else:
                target_batch = [target_point[x[1]] for x in history]

            # compute divergence
            action_batch = torch.stack(action_batch)
            target_batch = torch.stack(target_batch)
            loss_batch   = kl_loss(action_batch, target_batch)

            # optimize
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            # clear the history
            history.clear()

            # logging and checkpointing
            log_step += 1
            loss_acc += loss_batch.item()
            if log_step % LOGGING_FREQ == 0:
                # get timing
                t2 = time.time()
                dt = t2 - t1
                t1 = t2


                # this is gonna get a bit messy for the sake of formatting
                total_string  = f"{total_step}" 
                update_string = f"{log_step}" 
                loss_string   = f"{(loss_acc / LOGGING_FREQ):.6f}" 
                reward_string = f"{(reward_acc / log_step):.4f}" 
                time_string   = f"{dt:.2f} secs"

                total_spacing  = " " * max(0, 12 - len(total_string))
                update_spacing = " " * max(0, 10  - len(update_string))
                loss_spacing   = " " * max(0, 12 - len(loss_string))
                reward_spacing = " " * max(0, 10  - len(reward_string))
                time_spacing   = " " * max(0, 14 - len(time_string))

                print(f"total step: {total_string}", end=total_spacing)
                print(f"update step: {update_string}", end=update_spacing)
                print(f"loss: {loss_string}", end=loss_spacing)
                print(f"reward: {reward_string}", end=reward_spacing)
                if log_step % CHECKPOINT_FREQ == 0:
                    print(f"time: {time_string}", end=time_spacing)
                    print("Checkpointing ... ", end='')
                    sys.stdout.flush()
                    torch.save(agent.state_dict(), path.format(log_step))
                    print("done.")
                else:
                    print(f"time: {time_string}")

                # reset stats
                loss_acc   = 0
                reward_acc = 0

            # reset
            if r == REWARD_LOSS:
                if GAME_SIZE_RANDOM:
                    game_width  = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
                    game_height = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
                else:
                    game_width  = GAME_SIZE
                    game_height = GAME_SIZE

                game = SnakeEnvironment(game_height, game_width)
    
    # finalize
    print("Export ...")
    torch.save(agent.state_dict(), path.format(log_step))

def replay(agent, last_step, verbose=True):
    for i in range(REPLAY_GAMES):
        print()
        if verbose:
            print(' '*14, f"GAME {i+1}")
            print(' ', '-'*32)
            print()
        
        lifetime = []

        if GAME_SIZE_RANDOM:
            game_width  = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
            game_height = npr.randint(GAME_SIZE_LOWER, GAME_SIZE_UPPER)
        else:
            game_width  = GAME_SIZE
            game_height = GAME_SIZE

        game   = SnakeEnvironment(game_height, game_width)
        reward = REWARD_DEFAULT
        state  = game.env()
        lifetime.append(state)

        state = torch.FloatTensor(state).view(1,1,game_height,game_width)
        if verbose:
            print(game)
            print()

        steps = 0
        score = 0        
        stagnant_counter = 0
        while reward != REWARD_LOSS:
            a = agent(state)
            amax = a.argmax().item()
            reward, state = game.step(amax)
            lifetime.append(state)

            state = torch.FloatTensor(state).view(1,1,game_height,game_width)
            if verbose:
                print(game)
                print()
            
            if reward == REWARD_DEFAULT:
                stagnant_counter += 1
                if stagnant_counter > (game_height - 2) * (game_width - 2):
                    if verbose:
                        print("        >  stagnation")
                    break
            else:
                stagnant_counter = 0
                if reward == REWARD_POINT:
                    score += 1
            steps += 1
            
        a = F.softmax(a, dim=0)

        if verbose:
            print()
            print( "      fatal action choices")
            print(f"     Up        ->    {a[MOVE_UP]    : .4f}")
            print(f"     Down      ->    {a[MOVE_DOWN]  : .4f}")
            print(f"     Left      ->    {a[MOVE_LEFT]  : .4f}")
            print(f"     Right     ->    {a[MOVE_RIGHT] : .4f}")
    
        def color_slide(slide):
            mask_empty = slide == MAP_EMPTY 
            mask_wall  = slide == MAP_WALL 
            mask_body  = slide == MAP_SNAKE_BODY 
            mask_head  = slide == MAP_SNAKE_HEAD 
            mask_apple = slide == MAP_APPLE 
            
            color = np.zeros([game_height, game_width, 3], dtype=np.uint8)
            color[mask_empty] = [30, 52,  68]
            color[mask_wall]  = [36, 58,  76]
            color[mask_body]  = [48, 110, 126]
            color[mask_head]  = [64, 128, 148]
            color[mask_apple] = [48, 148, 132]

            return np.kron(color, np.ones((30,30,1)))

        if not os.path.isdir('eval'):
            os.mkdir('eval')
        filename = "eval/steps_{}k_game_{}_size_{}x{}_score_{}.gif".format(
            last_step//1000, i+1, game_width, game_height, score
        )
        print(f"\nRendering gameplay to GIF: {filename}")
        lifetime = [color_slide(slide) for slide in tqdm(lifetime)]
        print(f"Exporting ...\n")
        write_gif(lifetime, filename, fps=8)

def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = lambda *x, **y: parser.add_argument(*x, **y)
    add_arg('--mode',    type=str, metavar='MODE', help='train/eval', default="")
    add_arg('--weights', type=str, metavar='PATH', help='path to weights dict', default="")
    return parser.parse_args()

def walk(directory, ext):
    return [f for f in os.listdir(directory) if f.endswith(ext)]

if __name__ == "__main__":
    # CLI Arguments
    args = parse_args()
    bright = lambda s: f"{Style.BRIGHT}{s}{Style.RESET_ALL}"
    if 'train' not in args.mode.lower() and 'eval' not in args.mode.lower():
        print()
        print("Requires a valid mode to run. Consider 'train' or 'eval'.")
        print(f"{bright('$')} python train.py {bright('--mode train')} --weights /some/generic/path")
        print()
        exit()

    # check if the weight path is fake news
    if len(args.weights) == 0:
        print()
        print("A weights path is required for model export.")
        print(f"{bright('$')} python train.py --mode train {bright('--weights /some/generic/path')}")
        print()
        exit()

    # separate the directory out from the file, if needed
    if args.weights[-1] != '/':
        args.weights += '/'
    if not os.path.isdir(args.weights):
        print("The directory specified does not exist, would you like it to be created? ([Y]/n)")
        while True:
            ans = input().lower()
            if 'y' in ans or '\n' in ans or '\r' in ans or ans == '':
                os.mkdir(args.weights)
                break
            elif 'n' in ans:
                exit()
            else:
                print(f"Response invalid, try {bright('y')} or {bright('n')}")
                print()

    # creat the network
    agent = Agent()

    # see if theres any checkpoints in the folder
    last_step = 0
    files     = walk(args.weights, '.pt')
    values    = [int(x.split('-')[-1].split('.pt')[0]) for x in files]
    files     = [x for _, x in sorted(zip(values, files))]
    if len(files) != 0:
        print(f"Loading checkpoint: {files[-1]}")
        agent.load_state_dict(torch.load(args.weights + files[-1]))
        last_step = files[-1].split('-')[-1].split('.pt')[0]
        last_step = int(last_step)

    # run network
    if 'train' in args.mode:
        train(agent, args.weights + 'checkpoint-{}.pt', last_step)
    else:
        agent.eval()
        replay(agent, last_step, verbose=REPLAY_VERBOSITY)