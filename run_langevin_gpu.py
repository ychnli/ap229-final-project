import numpy as np
import math
from tqdm import tqdm
from itertools import combinations
from itertools import product
import time
import argparse
import torch
import os

SAVE_DIR = "/scratch/users/yucli/ap229_langevin"

def generate_signal(N, seed, device='cuda'):
    gen_data = torch.Generator(device=device).manual_seed(seed)
    x_star = torch.randn(N, generator=gen_data, device=device)
    x_star = x_star / x_star.norm() * torch.sqrt(torch.tensor(float(N), device=device))
    return x_star


def generate_init(N, seed, x_star, device='cuda'):
    # generates an initial vector orthogonal to the signal, such that the initial 
    # correlation is ~0 
    gen_x0 = torch.Generator(device=device).manual_seed(seed + 42) 
    x_star = x_star.to(device)
    r = torch.randn(N, generator=gen_x0, device=device)
    proj_coeff = torch.dot(r, x_star) / torch.dot(x_star, x_star)
    x_0 = r - proj_coeff * x_star
    x_0 = x_0 / torch.norm(x_0) * torch.sqrt(torch.tensor(float(N), device=device))
    return x_0


def generate_spiked_matrix_tensor(
        x_star: np.ndarray | torch.Tensor,
        Delta2: float,
        Delta_p: float,
        p: int = 3,
        dtype=torch.float32,
        device: str | None = None,
        seed: int | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if isinstance(x_star, np.ndarray):
        x_star = torch.tensor(x_star, dtype=dtype, device=device)
    else:
        x_star = x_star.to(dtype=dtype, device=device)

    N = x_star.shape[0]
    assert p >= 2, "tensor order p must be ≥2"

    Y_signal = torch.outer(x_star, x_star) / np.sqrt(N)
    noise_ut = torch.randn(N, N, dtype=dtype, device=device) * np.sqrt(Delta2)
    noise_ut = torch.triu(noise_ut, diagonal=1)
    Y = Y_signal + noise_ut + noise_ut.T
    Y.fill_diagonal_(0.)

    if p == 2:
        return Y

    if p != 3:
        raise NotImplementedError("Efficient symmetrisation is provided for p=3 only")

    coeff = np.sqrt(math.factorial(p - 1)) / (N ** ((p - 1) / 2))
    T_signal = coeff * torch.einsum('i,j,k->ijk', x_star, x_star, x_star)

    raw = torch.randn(N, N, N, dtype=dtype, device=device) * np.sqrt(Delta_p)
    perms = [(0, 1, 2), (1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0), (2, 0, 1)]
    T_noise = sum(raw.permute(p) for p in perms) / np.sqrt(6)

    T = T_signal + T_noise
    return Y, T


def langevin_dynamics_projected_gpu(x0, Y, T, delta2, deltap, steps=1000, step_size=0.01, device='cuda', seed=None):
    """
    Runs the Langevin algorithm for p = 3 on GPU
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = x0.shape[0]
    x = x0.clone().to(device)
    x = x / torch.norm(x) * torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))
    traj = [x.clone()]

    tensor_coeff = math.sqrt(2.0) / (deltap * N)

    for _ in tqdm(range(steps), desc="Langevin algorithm progress", unit='step'):
        grad_matrix = -1.0 / delta2 * (Y @ x) / torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))

        w = torch.einsum("ijk,j,k->i", T, x, x)
        grad_tensor = -tensor_coeff * w

        grad = grad_matrix + grad_tensor
        noise = torch.randn_like(x)
        x = x - step_size * grad + torch.sqrt(torch.tensor(2.0 * step_size, device=device)) * noise
        x = x / torch.norm(x) * torch.sqrt(torch.tensor(N, dtype=torch.float32, device=device))
        traj.append(x.clone())

    return torch.stack(traj)


def save_trajectory(traj, save_dir, filename="langevin_traj.pt", verbose=True):
    """
    Save the Langevin trajectory tensor to a file.

    Args:
        traj (torch.Tensor): Trajectory tensor of shape (steps+1, N)
        save_dir (str): Directory to save the file
        filename (str): Name of the saved file 
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(traj, filepath)

    if verbose:
        print(f"Trajectory saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run Langevin dynamics across noise regimes.")

    parser.add_argument("--N", type=int, default=100, help="Dimensionality of the problem")
    parser.add_argument("--M", type=int, default=5, help="Number of trials per noise setting")
    parser.add_argument("--steps", type=int, default=10000, help="Number of Langevin steps")
    parser.add_argument("--step_size", type=float, default=0.01, help="Langevin step size")
    parser.add_argument("--initial_seed", type=int, default=0, help="Initial seed (integer)")
    parser.add_argument("--output_dir", type=str, default=SAVE_DIR, help="Directory to save outputs")

    args = parser.parse_args()
    N, M, steps, step_size = args.N, args.M, args.steps, args.step_size
    init_seed = args.initial_seed
    output_dir = args.output_dir

    inv_delta2_vals = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25])
    delta2_vals = 1.0 / inv_delta2_vals
    deltap_vals = np.array([0.25, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25])

    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")
    for m in range(init_seed, init_seed+M):
        for i, delta2 in enumerate(delta2_vals):
            for j, deltap in enumerate(deltap_vals):
                print(f"Trial {m}: Langevin dynamics on Δ2 = {delta2}, Δp = {deltap} for N={N}...")

                seed_data = 1000 * m + i * 10 + j
                seed_langevin = 1000 * m + i * 10 + j + 5
                
                # generate signal 
                x_star = generate_signal(N, seed_data, device)
                
                # generate data
                Y, T = generate_spiked_matrix_tensor(x_star, delta2, deltap, p=3, seed=seed_data)
                
                # generate x_0
                x0 = generate_init(N, seed_data, x_star, device=device)
                
                init_corr = torch.dot(x0, x_star) / N
                print("initial correlation:", init_corr.item())  

                traj = langevin_dynamics_projected_gpu(x0, Y, T, delta2, deltap, steps, step_size, device=device, seed=seed_langevin)

                corr = torch.einsum('ij,j->i', traj, x_star) / N

                tag = f"N{N}_D2{delta2:.2f}_Dp{deltap:.2f}_trial{m}"
                torch.save({
                    'trajectory': traj.cpu(),
                    'signal': x_star.cpu(),
                    'correlation': corr.cpu()
                }, os.path.join(output_dir, f"langevin_{tag}.pt"))
                print('done!')

if __name__ == "__main__":
    main()     