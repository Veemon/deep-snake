# standard
import os
import sys
import random

from math import exp

# third party
import gym
from pyneat import pyneat

def distribution(x):
    return exp(-5 * (x / int(population_size * cutoff)))

def fitness(self):
    # initialize environment
    env = gym.make('CartPole-v0')
    s = env.reset()

    # the longest balanced is the most fit
    done = False
    while done == False:
        y = self.network.forward(s)
        a = y.index(max(y))
        s, r, done, _ = env.step(a)
        self.fitness += 1

if __name__ == '__main__':
    # Experiment Params
    input_nodes = 4
    output_nodes = 2

    population_size = 1000
    num_generations = 10

    # Population cutoff percentage
    cutoff = 0.75

    # Percentage chance to mutate
    mutation = 0.15

    # Constants used in speciation
    c1 = 1
    c2 = 1
    c3 = 1

    # Create the gene pool
    gene_pool = pyneat.GenePool(population_size,
                         num_generations,
                         cutoff,
                         mutation,
                         [c1,c2,c3],
                         logging=1,
                         num_threads=4)

    # Initialise the gene pool
    gene_pool.init(input_nodes, output_nodes, fitness, distribution)

    # Run an evolutionary period
    gene_pool.evolve()
