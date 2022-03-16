"""
--------------------------------------
Test of generate_population() function

Status: FINISHED
--------------------------------------
"""

import numpy as np
import pandas as pd


from Models.rvGA_model.rvga_functions import generate_population

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test
population_test = 25

# Mean - Variance | Mean + Variance
variance_lower_bound: float = .8
variance_higher_bound: float = 7.3

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))
test_population = generate_population(test_dataframe_1, latent_vector_length, population_test)

assert len(test_population) == population_test
assert len(test_population[0]) == latent_vector_length
