### Empirical verification of phase transitions in the Langevin algorithm for high dimensional Bayesian inference in the spiked 2 + 3-spin model
This is a mini repo for reproducing some of the main results in Mannelli et al. (2020) [[https://iris.uniroma1.it/retrieve/handle/11573/1472273/2012359/Sarao%20Mannelli_Marvels%20and%20Pitfalls%20of%20the%20Langevin_2020.pdf]]

* `run_langevin_gpu.py` is the main script for running the Langevin algorithm.
* `analysis.ipynb` is where the figures are generated.
* `compute_hamiltonian.py` is an auxiliary script for computing the Hamiltonian for Langevin trajectories after they have been computed (for compute reasons we don't explicitly compute the Hamiltonian during Langevin sampling)
