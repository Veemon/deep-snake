# deep-snake 

## Description 
This an experiment using deep q-networks to play the game of snake.
Initially this was done so tested on cartpole for model validation.
Special thanks to https://github.com/apaszke who wrote the base dqn tutorial for pytorch.
You can read it here http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Network 
The model used is a combination of the dueling variation, as well as the double learning method.
It is comprised of 3 convolutional layers, followed by 2 linear layers for the respective 
value and advantage paths.

## Dependencies 
* pytorch
* pygame
* colorama
* matplotlib
* (and a few others)

## CLI Arguments
To play snake run snake.py. 

| Argument        | Example       | Description                                                           |
| ----------------|:-------------:| ---------------------------------------------------------------------:|
| size={SIZE}     | size=32       | This changes the map size of the game.                                |
| ai={VERSION}    | ai            | This activates the agent mode.                                        |
|                 | ai=2000       | The following number represents the trained episode version to load.  |

## Results 
*in progress*
