"""
--------------------------------------
Test of generate_individual() function

Status: finished
--------------------------------------
"""

import pandas as pd
import numpy as np


from Models.rvGA_model.rvga_functions import generate_individual

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test

# Mean - Variance | Mean + Variance
variance_lower_bound: float = .8
variance_higher_bound: float = 7.3

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))
test_individual = generate_individual(test_dataframe_1, latent_vector_length)

assert len(test_individual) == N_test
assert all(variance_lower_bound < element < variance_higher_bound for element in test_individual) is True
