import random

from Models.rvGA_model.rvga_classes import *
from Models.rvGA_model.rvga_variables import max_length


def generate_individual():
    return Individual([random.randint(0, 1) for _ in range(max_length)])


def generate_population(n=0):
    return list([generate_individual() for _ in range(n)])


def individual_fitness(individual):
    return sum(individual),


def gene_importance():
    pass


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
