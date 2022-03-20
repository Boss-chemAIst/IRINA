"""
--------------------------------------
Test of make_crossover() function

Status: FINISHED
--------------------------------------
"""

import pandas as pd
import numpy as np

from Models.rvGA_model.rvga_functions import make_crossover, generate_individual, gene_importance_correction

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test
population_test = 25

# Mean - Variance | Mean + Variance
variance_lower_bound: float = .8
variance_higher_bound: float = 7.3

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))

# Crossover variables
crossover_rate = 100
gene_importance = gene_importance_correction(latent_vectors_df=test_dataframe_1,
                                             max_diff_in_mating=10)

parent_test_1 = generate_individual(latent_vectors_df=test_dataframe_1,
                                    latent_vector_length=latent_vector_length)

parent_test_2 = generate_individual(latent_vectors_df=test_dataframe_1,
                                    latent_vector_length=latent_vector_length)

print(parent_test_1)
print(parent_test_2)

make_crossover(parent_test_1, parent_test_2, gene_importance, crossover_rate)

print(parent_test_1)
print(parent_test_2)
print()
print(gene_importance)
