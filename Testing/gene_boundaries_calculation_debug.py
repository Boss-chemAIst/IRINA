"""
--------------------------------------
Test of gene_boundaries_calculation() function

Status: FINISHED
--------------------------------------
"""

import pandas as pd
import numpy as np

from Models.rvGA_model.rvga_functions import gene_boundaries_calculation

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test

# Mean - Variance | Mean + Variance
variance_lower_bound: float = .8
variance_higher_bound: float = 7.3

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))

gene_boundaries = gene_boundaries_calculation(latent_vectors_df=test_dataframe_1,
                                              latent_vector_length=latent_vector_length)
