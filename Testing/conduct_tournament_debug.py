"""
--------------------------------------
Test of conduct_tournament() function

Status: NOT STARTED
--------------------------------------
"""

from Models.rvGA_model.rvga_functions import conduct_tournament, generate_population
from Models.rvGA_model.rvga_variables import population_size
from Models.VAE_model.run_vae_model import vae_latent_vector_df, vae_latent_vector_length

test_population_1 = generate_population(latent_vectors_df=vae_latent_vector_df,
                                        latent_vector_length=vae_latent_vector_length,
                                        population_size=population_size)

offspring = conduct_tournament(population=test_population_1,
                               population_length=len(test_population_1),
                               survival_rate=0.5)
