"""
--------------------------------------
Test of conduct_tournament() function

Status: FINISHED
--------------------------------------
"""

import pandas as pd
import numpy as np

from Models.rvGA_model.rvga_functions import conduct_tournament, generate_population

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test

# Mean - Variance | Mean + Variance
variance_lower_bound: float = .8
variance_higher_bound: float = 7.3

population_size = 25
survival_rate = 0.2

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))

test_population_1 = generate_population(latent_vectors_df=test_dataframe_1,
                                        latent_vector_length=latent_vector_length,
                                        population_size=population_size)

print(len(test_population_1))

offspring = conduct_tournament(population=test_population_1,
                               population_length=len(test_population_1),
                               survival_rate=survival_rate)

print(len(offspring))
