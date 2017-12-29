# deep-snake 

## Description 
This an experiment using deep q-networks to play the game of snake.
Initially this was done so tested on cartpole for model validation.
Special thanks to https://github.com/apaszke who wrote the base dqn tutorial for pytorch.
You can read it here http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

## Network 
The model used is a combination of the dueling variation, as well as the double learning method.
Both primary and target networks are comprised of 3 convolutional layers, followed by 2 linear layers for the respective 
value and advantage paths.

## Dependencies 
* pytorch
* pygame
* matplotlib
* (and a few others)

## CLI Arguments
To play snake, run snake.py. 

| Argument        | Example       | Description                                                           |
| ----------------|:-------------:| ---------------------------------------------------------------------:|
| --size={SIZE}   | --size=32     | This changes the map size of the game.                                |
| --ai={VERSION}  | --ai          | This activates the agent mode.                                        |
|                 | --ai=2000     | The following number represents the trained episode version to load.  |
|                 | --ai=2k       | Alternatively, k will be parsed as a thousand.                        |
|                 | --ai=2m       | Likewise m will be parsed as a million.                               |
| --debug         | --debug       | This triggers a blocking a mode and prints debug information on death.|
| --path={PATH}   | --path=saves  | This points the save loader to a directory to load from               |

To train the agent, run train.py.

| Argument        | Example       | Description                                                           |
| ----------------|:-------------:| ---------------------------------------------------------------------:|
| --exploit       | --exploit     | Switches to a low decaying epsilon. (Experimental)                    |
| --plot          | --plot        | Enables experiment plotting, reduces speed noticeably.                |

To analyse an agent's history, run analyse.py

| Argument        | Example       | Description                                                           |
| ----------------|:-------------:| ---------------------------------------------------------------------:|
| -i={INTERVAL}   | -i=1000       | Specify's the version interval for analysis.                          |
| -p={PATH}       | -p=saves      | The path to load histories from.                                      |
| -n={TRIALS}     | -n=100        | The number of trials to perform for a given agent version.            |
| --plot          | --plot        | To solely plot the analysis file of a given path.

## Results 
*in progress*
