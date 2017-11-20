#!/usr/bin/env python3

import sys
import time
import operator
import random
import math
import pygame

import numpy as np

from model import *

from colorama import Fore, Back, Style
from colorama import init
init(autoreset=True)

# INDEXES
SYSTEMS = 0
GAME = 1

# Constants
start_fill = 30
pixel_size = 25
offset = 0

map_size = 16

map_empty = 0
map_wall = 1
map_fruit = 2
map_snake = 3

up = 0
right = 1
down = 2
left = 3

def clamp(a, b):
    if a <= 0:
        return 0
    elif a <= b:
        return a
    else:
        return b

def initialise_map(game_map, size):
    for i in range(size*size):
        game_map.append(map_empty)

    # Top / Bottom
    for i in range(size):
        game_map[i] = map_wall
        game_map[((size*size)-1) - i] = map_wall

    # Sides
    for i in range(1, size - 1):
        game_map[i * size] = map_wall
        game_map[i * size + (size - 1)] = map_wall

def draw_map(screen, game_map, size, map_val):
    for i, tile in enumerate(game_map):

        if(tile == map_empty):
            color = (map_val, clamp(map_val + 5, 255), clamp(map_val + 15, 255))

        if(tile == map_wall):
            color = (clamp(map_val + 9, 255), clamp(map_val + 18, 255), clamp(map_val + 32, 255))

        x = int(i%size) * pixel_size
        y = int(i/size) * pixel_size

        pygame.draw.rect(screen, color, (x, y, pixel_size - offset, pixel_size - offset), 0)

def draw_snake(screen, snake_pos, length, velocity, t):
    val = int(abs(math.sin((length/3)*t)*10))
    for index, i in enumerate(snake_pos):
        body_val = ((length - index) * 10) - (val*5)
        if velocity != 0:
            col = (clamp(0 + body_val, 255), clamp(150 + body_val, 255), clamp(180 + body_val, 255))
        else:
            col = (clamp(0 + body_val, 100), clamp(150 + body_val, 150), clamp(180 + body_val, 180))
        x = i[0] * pixel_size
        y = i[1] * pixel_size
        pygame.draw.rect(screen, col, (x, y, pixel_size - offset, pixel_size - offset), 0)

def draw_fruit(screen, fruit_pos, velocity, t):
    val = int(abs(math.sin(4*t)*60))
    if velocity != 0:
        col = (clamp(0 + val, 255), clamp(190 + val, 255), clamp(118 + val, 255))
    else:
        col = (clamp(0 + val, 100), clamp(190 + val, 190), clamp(118 + val, 120))
    x = fruit_pos[0] * pixel_size
    y = fruit_pos[1] * pixel_size
    pygame.draw.rect(screen, col, (x, y, pixel_size - offset, pixel_size - offset), 0)

def draw_score(sqr_size, score, score_font, screen, velocity, t):
    score_size = score_font.size(score)
    brightness = 10
    if velocity != 0:
        color = (clamp(40 + (math.cos(t)*brightness), 255),
                clamp(50 + (math.cos(t)*brightness), 255),
                clamp(60 + (math.cos(t)*brightness), 255))
    else:
        color = (clamp(190 + (math.cos(t)*brightness), 255),
                clamp(200 + (math.cos(t)*brightness), 255),
                clamp(210 + (math.cos(t)*brightness), 255))
    score_render = score_font.render(score, 100, color)
    mid = sqr_size/2
    screen.blit(score_render, (mid - score_size[0]/2, mid - score_size[1]/2))

def draw_dead(sqr_size, n_labels, label_size, labels, screen):
    # positions
    x_centers = []
    y_centers = []
    for i in range(n_labels):
        x_centers.append(sqr_size/2 - (label_size[i][0]/2))
        y_centers.append(sqr_size/2 - (label_size[i][1]/2))


    # blit render
    for i in range(n_labels):
        if i == 0:
            screen.blit(labels[i], (x_centers[i], y_centers[i] - 135))
        elif i == 1:
            screen.blit(labels[i], (x_centers[i], y_centers[i] + (0.3*y_centers[i])))
        else:
            screen.blit(labels[i], (x_centers[i], y_centers[i] + (0.4*y_centers[i])))

def move_snake(snake_pos, velocity, direction, length):
    if velocity == 0:
        return

    for i in range(length-1, 0, -1):
        snake_pos[i][0] = snake_pos[i-1][0]
        snake_pos[i][1] = snake_pos[i-1][1]

    if direction == up:
        snake_pos[0][1] -= velocity

    if direction == down:
        snake_pos[0][1] += velocity

    if direction == right:
        snake_pos[0][0] += velocity

    if direction == left:
        snake_pos[0][0] -= velocity

def get_body_direction(snake):
    head = snake[0]
    body = snake[1]

    direction = list(map(operator.sub, head, body))
    if direction[0] < 0:
        return left
    elif direction[0] > 0:
        return right
    elif direction[1] < 0:
        return up
    elif direction[1] > 0:
        return down

def collide_wall(game_map, map_size, snake_pos, velocity):
    snake_head = snake_pos[0]
    snake_index = (snake_head[1] * map_size) + snake_head[0]
    tile = game_map[int(snake_index)]

    if tile == map_wall:
        return 0

    return velocity

def collide_self(snake, length, velocity):
    head = snake[0]
    for i in range(2, length):
        if snake[i] == head:
            return 0
    return velocity

def collide_fruit(snake_pos, length, fruit_pos, game_map, map_size):
    snake_head = snake_pos[0]
    if snake_head == fruit_pos:
        length = add_segment(snake_pos, length)
        return new_fruit(game_map, map_size, snake_pos, length), length
    return fruit_pos, length

def add_segment(snake_pos, length):
    snake_pos.append(snake_pos[-1][:])
    return length + 1

def new_fruit(game_map, map_size, snake, length):
    fruit_pos = [0,0]

    fruit_pos[0] = random.randint(1, map_size-1)
    fruit_pos[1] = random.randint(1, map_size-1)

    # snake checking
    coef = 2
    if map_size < length*2:
        coef = 3
    bound_l = length/coef
    bound_center = snake[int(bound_l)]
    if fruit_pos[0] > bound_center[0] - bound_l and fruit_pos[0] < bound_center[0] + bound_l:
        if fruit_pos[1] > bound_center[1] - bound_l and fruit_pos[1] < bound_center[1] + bound_l:
            return new_fruit(game_map, map_size, snake, length)

    # wall checking
    fruit_index = (fruit_pos[1] * map_size) + fruit_pos[0]
    tile = game_map[int(fruit_index)]
    if tile == map_wall:
        return new_fruit(game_map, map_size, snake, length)

    return fruit_pos

def collapse_data(game_map, fruit_pos, snake_pos):
    flattened = game_map[:]

    fruit_index = (fruit_pos[1] * map_size) + fruit_pos[0]
    flattened[int(fruit_index)] = map_fruit

    for i in range(len(snake_pos)):
        snake_index = (snake_pos[i][1] * map_size) + snake_pos[i][0]
        snake_decay = map_snake - (0.5/len(snake_pos))*i
        flattened[int(snake_index)] = snake_decay

    return np.asarray(flattened, dtype=np.float32)

def reset_states(game_map, map_size):

    args = ARGS()

    # Reset Snake
    init_pos = map_size/2
    for i in range(args.length):
        args.snake_pos.append([init_pos,init_pos])

    # Generate New Fruit
    args.fruit_pos = new_fruit(game_map, map_size, args.snake_pos, args.length)

    return args

def print_state(s):
    for i in range(map_size):
        for j in range(map_size):
            ts = " {0:.1f} ".format(s[(i*map_size) + j])

            # empty
            if s[(i*map_size) + j] == 0.0:
                print(Fore.WHITE + Style.DIM + ts, end='')

            # walls
            elif s[(i*map_size) + j] == 1.0:
                print(Fore.BLUE + Style.DIM + ts, end='')

            # fruit
            elif s[(i*map_size) + j] == 2.0:
                print(Fore.GREEN + Style.BRIGHT + ts, end='')

            # snake head
            elif s[(i*map_size) + j] == 3.0:
                print(Fore.WHITE + Style.BRIGHT + ts, end='')

            # snake body (& soul)
            else:
                print(Fore.WHITE + ts, end='')

        print()

class ARGS():
    def __init__(self):
        # Systems
        self.tick_rate = 0.007
        self.last_t = 0

        self.move_rate = 0.09
        self.last_move = 0
        self.last_length = 3

        self.frame_times = []
        self.t = 0

        # Game
        self.direction = up
        self.velocity = 1
        self.length = 3

        self.snake_pos = []
        self.fruit_pos = []


# AI specific imports
if __name__ == '__main__':
    ai = False
    version = 0
    agent = False
    for arg in sys.argv:
        if arg[0:2] == 'ai':
            ai = True
            try:
                version = int(arg[3:])
            except:
                pass

        if arg[0:4] == 'size':
            try:
                map_size = int(arg[5:])
            except:
                pass

    if ai == True:

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.nn.functional as F
        from torch.autograd import Variable
        import torchvision.transforms as T

        from PIL import Image


        def state_tensor(surface):
            # RGB Array -> Channels, Height, Width
            pixels = pygame.surfarray.array3d(surface).transpose((2, 0, 1))

            # Convert to float, rescare, convert to torch tensor
            # (this doesn't require a copy)
            pixels = np.ascontiguousarray(pixels, dtype=np.float32) / 255
            pixels = torch.from_numpy(pixels)

            # Resize, and add a batch dimension (BCHW)
            resize = T.Compose([T.ToPILImage(),
                                T.Scale(32, interpolation=Image.CUBIC),
                                T.ToTensor()])

            return resize(pixels).unsqueeze(0).type(torch.FloatTensor)

        agent = Agent(num_episodes=100, gamma=0.95)
        num = load_checkpoint("saves", agent, version=version)



def main(agent):

    # Game
    global map_size
    init_pos = map_size/2
    if init_pos != int(init_pos):
        map_size -= 1
        init_pos = map_size/2

    fill_val = start_fill

    game_map = []
    initialise_map(game_map, map_size)

    # Font
    score_font = pygame.font.Font("font/Monoton-Regular.ttf", 160)
    font_top = pygame.font.Font("font/Monoton-Regular.ttf", 90)
    font_bot = pygame.font.Font("font/Monoton-Regular.ttf", 40)
    label_text = ['dead', 'enter-retry', 'escape-exit']

    aa = 10000
    col_top = (200,230,245)
    col_bot = (190,210,225)

    n_labels = len(label_text)
    label_size = []
    labels = []
    for i in range(n_labels):
        if i == 0:
            label_size.append(font_top.size(label_text[i]))
            labels.append(font_top.render(label_text[i], aa, col_top))
        else:
            label_size.append(font_bot.size(label_text[i]))
            labels.append(font_bot.render(label_text[i], aa, col_bot))

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
    draw_score(sqr_size, str(length - 3), score_font, screen, velocity, t)
    draw_fruit(screen, fruit_pos, velocity, t)
    draw_snake(screen, snake_pos, length, velocity, t)

    pygame.display.flip()

    while 1:
        t0 = time.clock()

        # Grab state
        if agent != False:
            s = state_tensor(screen)

        # input
        body_direction = get_body_direction(game_args.snake_pos)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if agent == False:
                    if event.key == pygame.K_UP and body_direction != down:
                        direction = up
                    if event.key == pygame.K_RIGHT and body_direction != left:
                        direction = right
                    if event.key == pygame.K_DOWN and body_direction != up:
                        direction = down
                    if event.key == pygame.K_LEFT and body_direction != right:
                        direction = left

                if event.key == pygame.K_RETURN:
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

                if event.key == pygame.K_ESCAPE:
                    return

        # AI - action
        if agent != False:
            desired = agent.select_action(s)
            if desired == up and body_direction != down:
                direction = up
            if desired == right and body_direction != left:
                direction = right
            if desired == down and body_direction != up:
                direction = down
            if desired == left and body_direction != right:
                direction = left

        if(t >= last_t + tick_rate):
            if(t >= last_move + move_rate):
                # update
                move_snake(snake_pos, velocity, direction, length)
                velocity = collide_self(snake_pos, length, velocity)
                velocity = collide_wall(game_map, map_size, snake_pos, velocity)
                fruit_pos, length = collide_fruit(snake_pos, length, fruit_pos, game_map, map_size)

                # if growth
                if length - last_length > 0:
                    # increase speed
                    move_rate -= (move_rate * 0.05)
                    last_length = length

                # move - timing
                last_move = t

            # system - timing
            frame_times.append(t - (last_t))
            last_t = t

            # agent death
            if agent != False:
                if velocity == 0:
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

            # draw
            fill_val = clamp(clamp(start_fill-(t*start_fill), 255)+(math.cos(t)*2)+(start_fill), 253)
            screen.fill((fill_val,fill_val,fill_val))

            draw_map(screen, game_map, map_size, fill_val + 2)

            if velocity != 0:
                draw_score(sqr_size, str(length - 3), score_font, screen, velocity, t)
                draw_fruit(screen, fruit_pos, velocity, t)
                draw_snake(screen, snake_pos, length, velocity, t)

            # death notice
            elif agent == False:
                draw_fruit(screen, fruit_pos, velocity, t)
                draw_snake(screen, snake_pos, length, velocity, t)
                draw_score(sqr_size, str(length - 3), score_font, screen, velocity, t)
                draw_dead(sqr_size, n_labels, label_size, labels, screen)

            pygame.display.flip()

            # title
            if len(frame_times) >= 10:
                avg = sum(frame_times)/len(frame_times)
                pygame.display.set_caption('Snek | FPS: {}'.format(int(1/avg)), 'None')
                frame_times = []

        t1 = time.clock()
        dt = t1 - t0
        t += dt

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Snek', 'None')
    main(agent)
