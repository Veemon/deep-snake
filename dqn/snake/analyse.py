#!/usr/bin/env python3

import os
import re
import sys

from itertools import count
from pathlib import Path

from snake import *
from model import *
from train import state_tensor

import matplotlib.pyplot as plt


class Args:
    def __init__(self):
        self.interval = 1000
        self.path = 'saves'
        self.num_trials = 10
        self.plot = False

def parse_args():
    result = Args()
    for arg in sys.argv:
        # interval
        if arg[:2] == '-i':
            try:
                result.interval = int(arg[3:])
            except:
                print("incorrect format for interval specification.")
                print("expected an integer.")
                sys.exit()

        # path
        elif arg[:2] == '-p':
            result.path = arg[3:]
            result.path.replace('/','')

        # number of trials
        elif arg[:2] == '-n':
            try:
                result.num_trials = int(arg[3:])
            except:
                print("incorrect format for number of trials.")
                print("expected an integer.")
                sys.exit()

        # just plot
        elif arg[:6] == '--plot':
            result.plot = True

    return result

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def last_version(path):
    with open(path + '/analysis.txt', 'r') as f:
        data = f.read()

    # split into version
    info = data.split('\n')
    info = [x.split('\t') for x in info]
    info = [x for x in info if x != ['']]

    # parse info
    return int(info[-1][0])

def plot_history(path):
    with open(path + '/analysis.txt', 'r') as f:
        data = f.read()

    # split into version
    info = data.split('\n')
    info = [x.split('\t') for x in info]
    info = [x for x in info if x != ['']]

    # parse info
    for i, x in enumerate(info):
        info[i][0] = int(x[0])
        info[i][1] = [int(y) for y in x[1][1:-1].replace(' ', '').split(',')]

    # plot
    x = []
    max_ = []
    avg_ = []
    min_ = []
    for history in info:
        x.append(history[0])
        max_.append(max(history[1]))
        min_.append(min(history[1]))
        avg_.append(sum(history[1]) / len(history[1]))

    plt.plot(x, max_, 'r')
    plt.plot(x, avg_, 'g')
    plt.plot(x, min_, 'b')
    plt.show()

def save_history(path, history, version):
    with open(path + '/analysis.txt', 'a+') as f:
        f.write("{}\t{}\n".format(version,history))

def run(path, file_path, num_trials):
    # split arguments
    version = int(file_path.replace('epoch_', '').replace('.model',''))

    # initialize agent
    agent = Agent(final_epsilon=0.01, fixed_epsilon=True)
    num = load_checkpoint(path, agent, version=version)

    # statistics
    score_history = []

    # run some trials
    for trial in range(num_trials):

        # Game Constants
        global map_size
        init_pos = map_size/2
        if init_pos != int(init_pos):
            map_size -= 1
            init_pos = map_size/2

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

        for t in count():

            # get momentum
            body_direction = get_body_direction(game_args.snake_pos)

            # get action
            desired = agent.select_action(state_tensor(screen))
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

            # initial reward
            reward = 0

            # if growth
            if length - last_length > 0:
                reward = 1
                last_length = length

            # if death
            if velocity == 0:
                score_history.append(length - 3)
                break

            # draw
            fill_val = start_fill
            screen.fill((fill_val,fill_val,fill_val))

            draw_map(screen, game_map, map_size, fill_val + 2)
            draw_fruit(screen, fruit_pos, velocity, t, ai=True)
            draw_snake(screen, snake_pos, length, velocity, t, ai=True, reward=reward)

            pygame.display.flip()

    save_history(path, score_history, version)

def main(args):
    # check for just plot
    if args.plot == True:
        plot_history(args.path)
        sys.exit()

    # make sure user wants to do this
    counter = 0
    temp = Path(args.path + '/analysis.txt')
    if temp.is_file():

        # check to continue
        print('-'*32)
        print("An analysis file was detected: '{}/analysis.txt'.".format(args.path))
        print("Would you like to continue the analysis? (y/n): ", end='')
        res = input()
        res = res.lower()

        if res == 'y' or res == 'yes':
            counter = last_version(args.path)
        else:
            # check to start anew
            print("Would you like to start a new analysis? (y/n): ", end='')
            res = input()
            res = res.lower()

            # check if they dont mind deleting
            if res == 'y' or res == 'yes':
                print('\n---[WARNING] ---')
                print("Proceeding will result in the deletion of this file.")
                print("Would you like to proceed? (y/n): ", end='')
                res = input()
                res = res.lower()

                if res == 'y' or res == 'yes':
                    os.remove(temp)
                else:
                    sys.exit()
            else:
                sys.exit()

    # get all versions
    files = list(filter(lambda f: f[- (len('.model')):] == '.model', next(os.walk(args.path))[2]))
    files = natural_sort(files)
    versions = [int(x.replace('epoch_','').replace('.model','')) for x in files]

    # run trials based on specified interval
    todo = []
    for ver in versions:
        if ver >= (counter + args.interval):
            counter = ver
            todo.append(files[versions.index(ver)])
    for path in todo:
        run(args.path, path, args.num_trials)

    # plot results
    plot_history(args.path)

if __name__ == '__main__':
    args = parse_args()
    main(args)
