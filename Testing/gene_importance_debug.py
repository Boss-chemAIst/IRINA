"""
--------------------------------------
Test of gene_importance() function

Status: FINISHED
--------------------------------------
"""

import pandas as pd
import numpy as np

from Models.rvGA_model.rvga_functions import gene_importance

N_test: int = 16
M_test: int = 1000
latent_vector_length: int = N_test

# Mean - Variance | Mean + Variance
variance_lower_bound: float = -1.8
variance_higher_bound: float = 2.3

test_dataframe_1 = pd.DataFrame(np.random.uniform(variance_lower_bound, variance_higher_bound, [N_test, M_test]))
gene_importance_test = gene_importance(test_dataframe_1)

assert len(gene_importance_test) == latent_vector_length
assert all(0 < element <= 1 for element in gene_importance_test) is True
assert isinstance(gene_importance_test, list) is True
