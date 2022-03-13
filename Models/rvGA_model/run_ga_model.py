# Import of common libraries
import random

# Import of local modules
from Models.rvGA_model.individual_creator import generate_individual
from Models.rvGA_model.fitness_function import individual_fitness

# Definition of rvGA model parameters
population_size = 25
max_generations = 50
random_seed = 2022
random.seed(random_seed)

# Definition of classes


class FitnessMax:
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def population_creator(n=0):
    return list([generate_individual() for i in range(n)])


population = population_creator(n=population_size)
generation_count = 0
