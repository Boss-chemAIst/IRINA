import random
import numpy as np

from Models.rvGA_model.rvga_classes import *


def generate_individual(latent_vectors_df, latent_vector_length) -> object:
    """

    :param latent_vector_length: vector length out of VAE
    :param latent_vectors_df: N-dimensional vectors for M target meshes (M columns).

    :return: Individual object (N-dimensional vector).

    Status: FINISHED
    """

    mean_values = latent_vectors_df.mean(axis=1).tolist()
    std_values = latent_vectors_df.std(axis=1).tolist()

    int_vector = []
    for N in range(latent_vector_length):
        distribution_lower_bound = mean_values[N] - 1.5*std_values[N]
        distribution_higher_bound = mean_values[N] + 1.5*std_values[N]
        gene_value = random.uniform(distribution_lower_bound, distribution_higher_bound)
        int_vector.append(gene_value)

    return Individual(int_vector)


def generate_population(latent_vectors_df=None,
                        latent_vector_length=0,
                        population_size=0):
    """

    :param latent_vectors_df: Scalar. Vector length out of VAE.
    :param latent_vector_length: DataFrame. N-dimensional vectors for M target meshes (M columns).
    :param population_size: Scalar. Number of individuals to be generated.

    :return: List of lists with the number of individuals equal to population size.

    Status: FINISHED

    """
    return list([generate_individual(latent_vectors_df, latent_vector_length) for _ in range(population_size)])


def individual_fitness(individual):
    return sum(individual),


def gene_importance(latent_vectors_df):
    """

    :param latent_vectors_df: Scalar. Vector length out of VAE.

    :return: List of genes' importance (min variance / gene variance).

    Status: IN PROGRESS

    """
    gene_standard_deviation = latent_vectors_df.std(axis=1)
    gene_standard_deviation_list = gene_standard_deviation.tolist()
    gene_importance_list = np.min(gene_standard_deviation_list) / gene_standard_deviation_list
    return gene_importance_list


def mating():
    pass


def make_crossover(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


def make_mutation(mutant, ind_prob=0.01):
    for index in range(len(mutant)):
        if random.random() < ind_prob:
            mutant[index] = 0 if mutant[index] == 1 else 1


def conduct_tournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

            offspring.append(max([population[i1], population[i2], population[i3]],
                                 key=lambda ind: ind.fitness.values[0]))

    return offspring


def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind
