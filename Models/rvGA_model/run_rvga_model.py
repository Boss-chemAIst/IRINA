def run_genetic_algorithm(latent_vectors_target,
                          vae_latent_vector_length,
                          gene_boundaries,
                          gene_importance=None,
                          population_size=50,
                          max_generations=50,
                          spectrum_proximity_percentage=90,
                          percentage_of_survived=50,
                          prob_mutation=10,
                          prob_crossover=80):

    if gene_importance is None:
        gene_importance = [1] * vae_latent_vector_length

    # Import of common libraries
    import random
    import numpy as np

    # Import of local rvGA parameters
    from Models.rvGA_model.rvga_functions import individual_fitness, \
        clone, \
        conduct_tournament, \
        make_crossover, \
        make_mutation, \
        generate_population

    # Population generation working based on the data from VAE
    population = generate_population(latent_vectors_df=latent_vectors_target,
                                     latent_vector_length=vae_latent_vector_length,
                                     population_size=population_size)

    # Generate fitness values for every individual
    fitness_values = list(map(individual_fitness, population))

    # Fill individual objects with fitness values
    for individual, fitness_value in zip(population, fitness_values):
        individual.fitness.values = fitness_value

    # Defining metrics to track during rvGA
    max_fitness_values = []
    mean_fitness_values = []

    # Unpacking all fitness values from individual objects in population
    fitness_values = [individual.fitness.values[0] for individual in population]

    """
    --------------------------------
    GENETIC ALGORITHM IMPLEMENTATION
    --------------------------------
    """

    # Initialization of generations counter
    generation_count = 0

    # Continue until either spectrum proximity reaches sufficient percentage or number of generations equals maximal
    while max(fitness_values) < (spectrum_proximity_percentage / 100) and generation_count < max_generations:

        # Add a generation
        generation_count += 1

        # Picks top percentage_of_survived % of individuals from the population
        survived = conduct_tournament(population=population,
                                      population_length=len(population),
                                      survival_rate=percentage_of_survived / 100)

        # Transfers fitness values in individual objects of the offspring
        survived = list(map(clone, survived))

        fitness_values = [individual.fitness.values[0] for individual in survived]
        mating_prob_proportions = fitness_values / np.mean(fitness_values)

        # Crossover between the survived individuals (probability of mating depends on the fitness)
        indexes = range(len(survived))

        offspring = []
        while len(offspring) < len(population):
            child1 = random.choices(indexes, weights=mating_prob_proportions)[0]
            child2 = random.choices(indexes, weights=mating_prob_proportions)[0]
            if random.random() < prob_crossover / 100:
                make_crossover(parent1=survived[child1],
                               parent2=survived[child2],
                               gene_importance=gene_importance,
                               crossover_rate=prob_crossover)
                offspring.append(survived[child1])
                offspring.append(survived[child2])

        # Introduces mutations to the individuals in the offspring
        for mutant in offspring:
            if random.random() < prob_mutation / 100:
                make_mutation(mutant=mutant,
                              ind_prob=1.0 / vae_latent_vector_length,
                              gene_importance=gene_importance,
                              gene_boundaries=gene_boundaries)

        # Updates the fitness values of the individuals in the offspring after crossovers and mutations
        fresh_fitness_values = list(map(individual_fitness, offspring))
        for individual, fitness_value in zip(offspring, fresh_fitness_values):
            individual.fitness.values = fitness_value

        # Setting up a new population of parents
        population[:] = offspring

        # Unpacking fitness values for a new population
        fitness_values = [ind.fitness.values[0] for ind in population]

        # Tracking the intermediate results of rvGA
        max_fitness = max(fitness_values)
        mean_fitness = sum(fitness_values) / len(population)
        max_fitness_values.append(max_fitness)
        mean_fitness_values.append(mean_fitness)
        print(f"Generation {generation_count}: Max fitness = {max_fitness}, Mean fitness = {mean_fitness}")

        best_index = fitness_values.index(max(fitness_values))
        print("Best individual = ", *population[best_index], "\n")
