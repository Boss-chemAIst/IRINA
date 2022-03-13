# Import of common libraries
import random

# Import of local modules
from Models.rvGA_model.functions import individualCreator, \
                                        oneMaxFitness, \
                                        clone, \
                                        selTournament, \
                                        cxOnePoint, \
                                        mutFlipBit

from Models.rvGA_model.model_variables import population_size, \
                                              max_length, \
                                              max_generations, \
                                              prob_crossover, \
                                              prob_mutation

# Definition of population creator


def population_creator(n=0):
    return list([individualCreator() for _ in range(n)])


population = population_creator(n=population_size)
generation_count = 0

fitness_values = list(map(oneMaxFitness, population))

for individual, fitness_value in zip(population, fitness_values):
    individual.fitness.values = fitness_value

max_fitness_values = []
mean_fitness_values = []

fitness_values = [individual.fitness.values[0] for individual in population]

total_individuals = population_size

while max(fitness_values) < max_length and generation_count < max_generations:
    generation_count += 1
    offspring = selTournament(population, len(population))
    total_individuals += len(offspring)
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < prob_crossover:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < prob_mutation:
            mutFlipBit(mutant, indpb=1.0/max_length)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitness_value in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitness_value

    population[:] = offspring

    fitness_values = [ind.fitness.values[0] for ind in population]

    max_fitness = max(fitness_values)
    mean_fitness = sum(fitness_values) / len(population)
    max_fitness_values.append(max_fitness)
    mean_fitness_values.append(mean_fitness)
    print(f"Generation {generation_count}: Max fitness = {max_fitness}, Mean fitness = {mean_fitness}")

    best_index = fitness_values.index(max(fitness_values))
    print("Best individual = ", *population[best_index], "\n")
