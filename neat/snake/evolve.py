# standard
import os
import sys
import random

from snake import *
from pyneat import pyneat

def fitness(self):
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
	
	# While still alive
	num_movements = 0
	max_score = 0
	while velocity == 1:
		# Update movement counter
		num_movements += 1
        
        # Grab state
		s = collapse_data(game_map, fruit_pos, snake_pos)

		# Get momentum
		body_direction = get_body_direction(game_args.snake_pos)

		# Get action
		y = self.network.forward(s)
		desired = y.index(max(y))
		if desired == up and body_direction != down:
			direction = up
		if desired == right and body_direction != left:
			direction = right
		if desired == down and body_direction != up:
			direction = down
		if desired == left and body_direction != right:
			direction = left

		# Update
		move_snake(snake_pos, velocity, direction, length)
		velocity = collide_self(snake_pos, length, velocity)
		velocity = collide_wall(game_map, map_size, snake_pos, velocity)
		fruit_pos, length = collide_fruit(snake_pos, length, fruit_pos, game_map, map_size)

		# If growth
		if length - last_length > 0:
			max_score += 1
			last_length = length

	# Calculate fitness
	self.fitness = (max_score*100) - num_movements
	return False, self

if __name__ == '__main__':
    # Experiment Params
    input_nodes = map_size ** 2
    output_nodes = 4
	
    population_size = 100
    num_generations = 100

    # Population cutoff percentage
    cutoff = 0.35

    # Percentage chance to mutate
    weight_chance = 0.7
    structure_chance = 0.5

    # Constants used in speciation
    c1 = 1.0
    c2 = 1.0
    c3 = 0.4

    # Speciation threshold
    sigma_t = 2.0

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
