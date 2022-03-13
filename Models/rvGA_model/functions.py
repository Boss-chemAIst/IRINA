import random

from Models.rvGA_model.object_classes import *
from Models.rvGA_model.model_variables import max_length


def individualCreator():
    return Individual([random.randint(0, 1) for i in range(max_length)])


def populationCreator(n=0):
    return list([individualCreator() for i in range(n)])


def oneMaxFitness(individual):
    return sum(individual),


def gene_importance():
    pass


def mating():
    pass


def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


def selTournament(population, p_len):
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
