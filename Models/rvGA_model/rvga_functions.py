import random
import numpy as np

from Models.rvGA_model.rvga_classes import *


def generate_individual(latent_vectors_df, latent_vector_length) -> object:
    """

    Generates a single individual based on gene variations.

    :param latent_vector_length: vector length out of VAE
    :param latent_vectors_df: N-dimensional vectors for M target meshes (M columns).

    :return: Individual object (N-dimensional vector).

    Status: FINISHED
    """

    mean_values = latent_vectors_df.mean(axis=1).tolist()
    std_values = latent_vectors_df.std(axis=1).tolist()

    int_vector = []
    for N in range(latent_vector_length):
        distribution_lower_bound = mean_values[N] - 1.5 * std_values[N]
        distribution_higher_bound = mean_values[N] + 1.5 * std_values[N]
        gene_value = random.uniform(distribution_lower_bound, distribution_higher_bound)
        int_vector.append(gene_value)

    return Individual(int_vector)


def generate_population(latent_vectors_df=None,
                        latent_vector_length=0,
                        population_size=0):
    """

    Generates population of individuals of fixed size based on variations in genes.

    :param latent_vectors_df: Scalar. Vector length out of VAE.
    :param latent_vector_length: DataFrame. N-dimensional vectors for M target meshes (M columns).
    :param population_size: Scalar. Number of individuals to be generated.

    :return: List of lists with the number of individuals equal to population size.

    Status: FINISHED

    """

    return list([generate_individual(latent_vectors_df, latent_vector_length) for _ in range(population_size)])


def individual_fitness(individual):
    """

    Calculates the fitness of a single individual based on COMSOL calculations.

    :param individual: Vector of genes of a single individual. Sent to VAE to reconstruct COMSOL-readable geometry.

    :return: Metrics of individual's spectrum proximity to the desired one (goes from COMSOL simulation).

    Status: NOT STARTED

    """

    return sum(individual),


def gene_importance_correction(latent_vectors_df, max_mutation_increase):
    """

    Based on the VAE model results, grades every individual's gene by importance correlating with its standard
    deviation.

    :param max_mutation_increase: Stands for N in N-fold increased mutation rate for the least important gene.
    :param latent_vectors_df: Scalar. Vector length out of VAE.

    :return: List of genes' importance (min variance / gene variance).

    Status: FINISHED

    """

    gene_standard_deviation = latent_vectors_df.std(axis=1)

    gene_importance_relative = np.min(gene_standard_deviation) / gene_standard_deviation
    data = gene_importance_relative.tolist()

    data_range = np.max(data) - np.min(data)
    data_norm = [(i - np.min(data)) / data_range for i in data]

    custom_range = 1 - (1 / max_mutation_increase)
    gene_importance_corr = (np.array(data_norm) * custom_range) + (1 / max_mutation_increase)

    return gene_importance_corr.tolist()


def mating(population):
    """

    Implements crossovers within the population and mutates the offspring.

    :param population: Population to produce next generation.

    :return: Mutated offspring.

    Status: NOT STARTED

    """


def make_crossover(parent1, parent2):
    """

    Implements crossover between two parent individuals.

    :param parent1: Individual (first parent).
    :param parent2: Individual (second parent).

    :return: Two child.

    Status: NOT STARTED

    """

    child1, child2 = [], []

    return child1, child2


def make_mutation(mutant, ind_prob=0.01):
    """

    Introduces mutations to the offspring depending on the gene importance.

    :param mutant: Individual to be mutated.
    :param ind_prob: Standard probability of mutation.

    :return: Mutant individual.

    Status: NOT STARTED

    """

    for index in range(len(mutant)):
        if random.random() < ind_prob:
            mutant[index] = 0 if mutant[index] == 1 else 1


def conduct_tournament(population, population_length, survival_rate):
    """

    Filters population, top (survival_rate * 100) percent of individuals.

    :param population: List of lists with individuals in population.
    :param population_length: Number of individuals in the population.
    :param survival_rate: Rate of individuals surviving the tournament (from 0 to 1).

    :return: Offspring close to survival_rate * population_length.

    Status: FINISHED

    """

    offspring = []
    while len(offspring) <= np.floor(population_length * survival_rate):

        fitness_values = [ind.fitness.values[0] for ind in population]
        best = population[fitness_values.index(max(fitness_values))]
        offspring.append(best)

        best_index = population.index(best)
        del population[best_index]

    return offspring


def clone(individual):
    """

    Transfers fitness values.

    :param individual: Individual, which fitness values will be transferred to the same individual in the next
    generation.

    :return: Individual with written fitness value.

    Status: FINISHED

    """

    ind = Individual(individual[:])
    ind.fitness.values[0] = individual.fitness.values[0]
    return ind
