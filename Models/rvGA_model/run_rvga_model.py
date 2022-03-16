# Import of common libraries
import random


# Import of local rvGA parameters
from Models.rvGA_model.rvga_functions import individual_fitness, \
    clone, \
    conduct_tournament, \
    make_crossover, \
    make_mutation, \
    generate_population

from Models.rvGA_model.rvga_variables import population_size, \
    latent_vector_length, \
    max_generations, \
    prob_crossover, \
    prob_mutation


# Import of local VAE parameters
from Models.VAE_model.run_vae_model import vae_latent_vector_df, vae_latent_vector_length


# Definition of population creator, working based on the data from VAE
population = generate_population(latent_vectors_df=vae_latent_vector_df,
                                 latent_vector_length=vae_latent_vector_length,
                                 population_size=population_size)


# Generate fitness values for every individual
fitness_values = list(map(individual_fitness, population))

for individual, fitness_value in zip(population, fitness_values):
    individual.fitness.values = fitness_value

max_fitness_values = []
mean_fitness_values = []

fitness_values = [individual.fitness.values[0] for individual in population]


# Genetic algorithm implementation
total_individuals = population_size

generation_count = 0
while max(fitness_values) < latent_vector_length and generation_count < max_generations:
    generation_count += 1
    offspring = conduct_tournament(population, len(population))
    total_individuals += len(offspring)
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < prob_crossover:
            make_crossover(child1, child2)

    for mutant in offspring:
        if random.random() < prob_mutation:
            make_mutation(mutant, ind_prob=1.0 / latent_vector_length)

    freshFitnessValues = list(map(individual_fitness, offspring))
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
