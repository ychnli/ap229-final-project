import numpy as np
import math
from tqdm import tqdm
import xarray as xr
import re
import os
from itertools import product
import torch

from util import load_data, compute_hamiltonian
from run_langevin_gpu import generate_signal, generate_spiked_matrix_tensor

def main():
    corr_N1000 = load_data("/scratch/users/yucli/ap229_langevin/N1000_continued", var='correlation')
    traj_N1000 = load_data("/scratch/users/yucli/ap229_langevin/N1000_continued", var='trajectory')
    signal_N1000 = load_data("/scratch/users/yucli/ap229_langevin/N1000_continued", var='signal')

    device = "cuda" if torch.cuda.is_available() else "cpu"

    easy_subset = corr_N1000.isel(delta2=slice(0,3), deltap=slice(6,9))
    hard_subset = corr_N1000.isel(delta2=slice(3,6), deltap=slice(0,2))

    inv_delta2_vals = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25])
    delta2_vals = 1.0 / inv_delta2_vals
    deltap_vals = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25])

    energy_N1000 = xr.full_like(corr_N1000, fill_value=np.nan) 

    init_seed = 0
    N, M = 1000, 3 
    for i, delta2 in enumerate(delta2_vals):
        for j, deltap in enumerate(deltap_vals):
            delta2_rounded = np.round(delta2, 2)
            if delta2_rounded not in easy_subset.delta2 and delta2_rounded not in hard_subset.delta2:
                print(f"skipping Δ2 = {delta2}, Δp = {deltap}")
                continue
            if deltap not in easy_subset.deltap and deltap not in hard_subset.deltap:
                print(f"skipping Δ2 = {delta2}, Δp = {deltap}")
                continue

            for m in range(M):
                print(f"Computing Hamiltonian for Trial {m}: Langevin dynamics on Δ2 = {delta2}, Δp = {deltap} for N={N}...")

                seed_data = 1000 * m + i * 10 + j
                seed_langevin = 1000 * m + i * 10 + j + 5
                
                # generate signal 
                x_star = generate_signal(N, seed_data, device)
                
                # generate data
                Y, T = generate_spiked_matrix_tensor(x_star, delta2, deltap, p=3, seed=seed_data)
                
                # check that the signal here matches the one generated during training
                x_star_prev = torch.tensor(signal_N1000.sel(delta2=delta2_rounded, deltap=deltap, trial=m).values, dtype=torch.float32, device=device)
                assert(torch.allclose(x_star, x_star_prev, rtol=1e-05, atol=1e-08))
                
                traj = torch.tensor(traj_N1000.sel(delta2=delta2_rounded, deltap=deltap, trial=m).values, dtype=torch.float32, device=device)
                
                for t in tqdm(range(10001)):
                    energy = compute_hamiltonian(traj[t,:].unsqueeze(dim=0), Y, T, delta2, deltap)
                    energy_N1000.loc[dict(delta2=delta2_rounded, deltap=deltap, trial=m, step=t)] = energy.cpu().numpy()[0]
                
                tag = f"N{N}_D2{delta2:.2f}_Dp{deltap:.2f}_trial{m}"
                energy_N1000.sel(delta2=delta2_rounded, deltap=deltap, trial=m).to_netcdf(f"/scratch/users/yucli/ap229_langevin/N1000_continued/energy_{tag}.nc")
    
    energy_N1000.to_netcdf(f"/scratch/users/yucli/ap229_langevin/N1000_continued/energy.nc")

if __name__ == "__main__":
    main()