# standard
import os
import sys
import random

# third party
import gym
from pyneat import pyneat

def fitness(self):
    # initialize environment
    env = gym.make('CartPole-v0')
    s = env.reset()

    # the longest balanced is the most fit
    done = False
    while done == False:
        y = self.network.forward(s)
        a = 1 if y[0] > 0 else 0
        s, r, done, _ = env.step(a)
        self.fitness += 1

    return False, self

if __name__ == '__main__':
    # Experiment Params
    input_nodes = 4
    output_nodes = 1

    population_size = 1000
    num_generations = 100

    # Population cutoff percentage
    cutoff = 0.45

    # Percentage chance to mutate
    weight_chance = 0.8
    structure_chance = 0.4

    # Constants used in speciation
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4

    # Speciation threshold
    sigma_t = 3.0

    # Create the gene pool
    gene_pool = pyneat.GenePool(population_size,
                         num_generations,
                         cutoff,
                         [c1,c2,c3],
                         sigma_t,
                         logging=1,
                         num_threads=8)

    # Initialise the gene pool
    gene_pool.init(input_nodes, output_nodes, fitness)

    # Run an evolutionary period
    gene_pool.evolve(weight_chance, structure_chance)