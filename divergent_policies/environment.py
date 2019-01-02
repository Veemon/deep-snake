from colorama import Fore, Style, init

import numpy as np
import numpy.random as npr

init()

MAP_EMPTY      = 0
MAP_WALL       = 1
MAP_SNAKE_BODY = 2
MAP_SNAKE_HEAD = 3
MAP_APPLE      = 4 

MOVE_UP    = 0
MOVE_DOWN  = 1
MOVE_LEFT  = 2
MOVE_RIGHT = 3

REWARD_LOSS    = -1
REWARD_DEFAULT =  0
REWARD_POINT   =  1

DIRECTION_DICT = {
    MOVE_UP    : np.array([-1,  0]),   
    MOVE_DOWN  : np.array([ 1,  0]), 
    MOVE_LEFT  : np.array([ 0, -1]), 
    MOVE_RIGHT : np.array([ 0,  1]),
}

COLOR_DICT = {
    str(MAP_EMPTY)      : ' ',
    str(MAP_WALL)       : Style.NORMAL + Fore.WHITE  + str(MAP_WALL),
    str(MAP_SNAKE_BODY) : Style.NORMAL + Fore.YELLOW + str(MAP_SNAKE_BODY),
    str(MAP_SNAKE_HEAD) : Style.BRIGHT + Fore.YELLOW + str(MAP_SNAKE_HEAD),
    str(MAP_APPLE)      : Style.BRIGHT + Fore.RED    + str(MAP_APPLE),
}

class SnakeEnvironment:
    def __init__(self, h, w=None):
        # if no width make it a square
        if w is None:
            w = h

        # construction
        self.dead      = False
        self.grow      = False
        self.direction = MOVE_UP
        self.map       = np.zeros([h,w], dtype=np.uint8)
        self.apple     = np.zeros([2], dtype=np.uint8)
        self.snake = [
            np.array([(h // 2) + 0, w // 2]),
            np.array([(h // 2) + 1, w // 2]),
            np.array([(h // 2) + 2, w // 2]),
        ]

        # build the wall(s)
        self.map[ 0,:] = MAP_WALL
        self.map[-1,:] = MAP_WALL
        self.map[:, 0] = MAP_WALL
        self.map[:,-1] = MAP_WALL

        # new apple
        self.new_apple()

    def step(self, action):
        # if you're dead you cant move (theoretically)
        if self.dead:
            return REWARD_LOSS, self.env()

        # disregard reversal actions
        intention = action
        prevented = False
        for a, b in [
            [MOVE_DOWN, MOVE_UP],
            [MOVE_LEFT, MOVE_RIGHT],
        ]:
            if (self.direction == a and action == b) or (self.direction == b and action == a):
                intention = self.direction
                prevented = True
                break
        if not prevented:
            self.direction = intention
        
        # save for when we need to grow
        if self.grow:
            growth_pos = self.snake[-1].copy()

        # body has to follow
        for i in reversed(range(len(self.snake))):
            if i != 0:
                self.snake[i] = self.snake[i-1].copy()

        # move snake head
        self.snake[0] += DIRECTION_DICT[intention]

        # time to grow
        if self.grow:
            self.snake.append(growth_pos)
            self.new_apple()
            self.grow = False

        # setup rewards - welcome to RL baby
        reward = REWARD_DEFAULT

        # collisions - self
        for segment in self.snake[1:]:
            if (self.snake[0] == segment).all():
                self.dead = True
                reward = REWARD_LOSS

        # collisions - wall
        if reward == REWARD_DEFAULT:
            if self.map[tuple(self.snake[0])] == MAP_WALL:
                self.dead = True
                reward = REWARD_LOSS

        # collisions - jabłko
        if reward == REWARD_DEFAULT:
            if (self.snake[0] == self.apple).all():
                self.grow = True
                reward = REWARD_POINT

        # if nothing happened, thats okay too
        return reward, self.env()

    def env(self):
        temp = self.map.copy()
        if self.apple.sum() != 0:
            temp[tuple(self.apple)] = MAP_APPLE
        for i, (y, x) in enumerate(self.snake):
            if i == 0:
                temp[y,x] = MAP_SNAKE_HEAD
            else:
                temp[y,x] = MAP_SNAKE_BODY
        return temp
    
    def new_apple(self):
        # see this apply is szmart because it always placed in a valid position,
        # unless there's no where to put it :O
        temp = self.env()
        x, y = np.where(temp == MAP_EMPTY)
        if len(x) > 0:
            c = npr.randint(0, len(x))
            self.apple = np.array([x[c], y[c]]) 
        else:
            self.apple = np.array([0, 0]) 

    def __str__(self):
        # you have to be very szmart to understand this codes
        string = str(self.env())
        string = string.replace('[[', ' [').replace(']]', ']')
        string = string.replace(' [', '').replace(']', '')

        # add colors to the string
        string = list(string)
        for i in reversed(range(len(string))):
            if string[i] != ' ' and string[i] != '\n':
                string.insert(i, COLOR_DICT[string[i]] + Style.RESET_ALL)
                del string[i+1]

        return "".join(string)


# run this file to play the game step by step
if __name__ == "__main__":
    import os, sys
    if os.name == 'nt':
        import msvcrt
        get_input = msvcrt.getch

    env = SnakeEnvironment(10)
    print(env)

    last_action = 0
    reward      = REWARD_DEFAULT
    while reward != REWARD_LOSS:
        print()

        ans = get_input().decode('utf-8')
        if 'w' in ans:
            ans = 0
        elif 'a' in ans:
            ans = 2
        elif 's' in ans:
            ans = 1
        elif 'd' in ans:
            ans = 3

        if ans == '' or ans == '\n' or ans == '\r':
            ans = last_action
        else:
            last_action = ans 

        reward, state = env.step(ans)
        print(env)
        sys.stdout.flush()
