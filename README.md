--------------------------------------------------------------------
Intrasystemic Relations-Informed Neuroevolutionary Algorithm (IRINA)
--------------------------------------------------------------------
--------------------------------------------------------------------

Description:
------------

IRINA extracts system-specific information using variational autoencoder (VAE) architecture and uses it to bias a real-valued genetic algorithm (rvGA) towards an exploration of the particular system design sub-spaces.

Features:
---------

- VAE reduces the system complexity to N-dimensional vectors
- Variance of each gene in the vector corresponds to 1/importance
- Boundaries of each gene determine the sub-space to be explored
- rvGA is biased to explore this sub-space via importance-based individual mating and mutations
- Severity, frequency, and direction of mutations depends on the gene importance and boundaries
- Dominance in mating depends on the fitness
- Crossover rate depends on the gene importance

How to:
-------

- Check if COMSOL and Matlab is installed
- Add '...\IRINA\Python_COMSOL_connection\software\matlab_connection\mfiles' to Matlab search path 
  with addpath() function
- 