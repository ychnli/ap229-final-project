import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize_scalar
from tqdm import tqdm
import math

# =====================================
# 1) Data generation functions (Corrected)
# =====================================

def generate_spiked_matrix_tensor(N, Delta2, Delta_p, p=3, seed=None):
    """
    Generate a random spike x_star (||x_star|| = sqrt(N)),
    and noisy observations Y (matrix) and T (3-order tensor).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Generate Normalized True spike x_star
    x_star = np.random.randn(N)
    x_star *= np.sqrt(N) / np.linalg.norm(x_star)

    # 2) Noisy matrix Y = signal + noise
    signal_mat = np.outer(x_star, x_star) / np.sqrt(N)
    # BUG FIX 1: Correctly generate symmetric noise with variance Delta2
    # for off-diagonal elements.
    noise_mat = np.random.randn(N, N) * np.sqrt(Delta2)
    noise_mat = np.triu(noise_mat, 1)
    noise_mat = noise_mat + noise_mat.T
    Y = signal_mat + noise_mat

    # 3) Noisy tensor T (order-p) = signal + noise, fully symmetric
    T = np.zeros((N, N, N))
    # BUG FIX 2: Correct scaling factor with factorial for tensor signal
    scale = np.sqrt(math.factorial(p - 1)) / (N ** ((p - 1) / 2.0))
    for i, j, k in combinations(range(N), 3):
        sig = scale * x_star[i] * x_star[j] * x_star[k]
        noise = np.random.randn() * np.sqrt(Delta_p)
        val = sig + noise
        # assign to all permutations
        T[i, j, k] = T[i, k, j] = T[j, i, k] = T[j, k, i] = T[k, i, j] = T[k, j, i] = val

    return x_star, Y, T

# =====================================
# 2) Vector AMP for spiked matrix+tensor (p=3) (Corrected)
# =====================================

def run_amp_state_evolution(Delta2, Delta_p, p=3, max_iter=300, tol=1e-7):
    """
    Run state evolution for AMP. Determines if AMP can find the signal.
    """
    # BUG FIX 3: Initialize with a small positive overlap, not a random one.
    m_t = 1e-6

    for _ in range(max_iter):
        m_prev = m_t
        m_t = 1 - 1 / (1 + m_prev / Delta2 + (m_prev**(p - 1)) / Delta_p)
        if abs(m_t - m_prev) < tol:
            break

    # Succeeds if the final overlap is significantly greater than 0
    return m_t > 1e-4

# =====================================
# 3) Discretized Langevin on the sphere (p=3)
# =====================================

def run_langevin(Y, T, x_star, Delta2, Delta_p, p=3, dt=1e-2, max_steps=15000, threshold=0.5):
    """
    Run discretized Langevin dynamics on the sphere for p=3.
    """
    N = x_star.shape[0]
    x = np.random.randn(N)
    x *= np.sqrt(N) / np.linalg.norm(x)

    for step in range(max_steps):
        # Gradient of Hamiltonian from Eq. (4)
        grad_H_matrix = (1.0 / (Delta2 * np.sqrt(N))) * (Y @ x)
        w = np.einsum('ijk,j,k->i', T, x, x)
        grad_H_tensor = (np.sqrt(math.factorial(p - 1)) / (Delta_p * N**((p - 1) / 2.0))) * w
        grad_H = grad_H_matrix + grad_H_tensor

        # Euler step + noise
        noise = np.random.randn(N) * np.sqrt(2 * dt)
        x_temp = x + dt * grad_H + noise

        # Project back to sphere
        x = np.sqrt(N) * x_temp / np.linalg.norm(x_temp)

        m = np.dot(x, x_star) / N
        if m >= threshold:
            return True # Success
            
    return False # Failure

# =====================================
# 4) Full grid simulation
# =====================================

def simulate_phase_diagram(N, inv_delta2_vals, delta_p_vals, p=3, trials=10):
    H = len(inv_delta2_vals)
    W = len(delta_p_vals)
    phase_amp = np.zeros((W, H), dtype=int)
    langevin_hard = np.zeros((W, H), dtype=bool)

    progress = tqdm(total=H * W, desc="Simulating Phase Diagram")

    for i, inv_d2 in enumerate(inv_delta2_vals):
        d2 = 1.0 / inv_d2
        for j, dp in enumerate(delta_p_vals):
            # --- Bayesian Possibility (Information-Theoretic) ---
            # Determined by the global maximizer of Φ_RS(m) from Eq. (5)
            def neg_phi_rs(m):
                if m <= 1e-9 or m >= 1 - 1e-9: return np.inf
                term1 = 0.5 * np.log(1 - m)
                term2 = m / 2
                term3 = (m**2) / (4 * d2)
                term4 = (m**p) / (2 * p * dp)
                return -(term1 + term2 + term3 + term4)

            res = minimize_scalar(neg_phi_rs, bounds=(1e-9, 1 - 1e-9), method='bounded')
            m_star = res.x if res.fun != np.inf and res.x > 1e-5 else 0
            bayes_possible = m_star > 0

            # --- AMP Performance ---
            amp_succeeds = run_amp_state_evolution(d2, dp, p)

            # --- Langevin Performance ---
            langevin_succeeds = False
            if amp_succeeds: # Only test Langevin where AMP is expected to work
                langevin_success_count = 0
                for trial in range(trials):
                    # Use different seeds for Langevin to get an independent statistical sample
                    x_star, Y, T = generate_spiked_matrix_tensor(N, d2, dp, p, seed=trial + trials)
                    if run_langevin(Y, T, x_star, d2, dp, p):
                        langevin_success_count += 1
                if langevin_success_count / trials > 0.5:
                    langevin_succeeds = True

            # --- Classify Phases ---
            if not bayes_possible:
                phase_amp[j, i] = 0  # Impossible
            elif not amp_succeeds:
                phase_amp[j, i] = 1  # AMP-Hard
            else: # bayes_possible and amp_succeeds
                phase_amp[j, i] = 2  # AMP-Easy
                if not langevin_succeeds:
                    langevin_hard[j, i] = True # Langevin-Hard is a subset of AMP-Easy
            
            progress.update(1)

    progress.close()
    return phase_amp, langevin_hard


# =====================================
# 5) Example usage
# =====================================

if __name__ == "__main__":
    N = 80
    p = 3
    # Use axis ranges similar to the paper's figures
    inv_delta2_vals = np.linspace(0.4, 2.2, 8)
    delta_p_vals = np.linspace(0.4, 3.0, 8)
    
    phase_amp, langevin_hard = simulate_phase_diagram(N, inv_delta2_vals, delta_p_vals, p=p, trials=5)
    
    # =====================================
    # 6) Plotting the results
    # =====================================
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    cmap = plt.cm.colors.ListedColormap(['#C00000', '#FFFFAA', '#228B22']) # Red, Yellow, Green
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # Use pcolormesh for better grid alignment
    im = ax.pcolormesh(delta_p_vals, inv_delta2_vals, phase_amp.T, shading='auto', cmap=cmap, norm=norm)

    # Colorbar
    cbar = fig.colorbar(im, ticks=[0, 1, 2], spacing='proportional')
    cbar.ax.set_yticklabels(['Impossible', 'AMP-Hard', 'AMP-Easy'])

    # Overlay Langevin-hard markers
    y_coords, x_coords = np.where(langevin_hard.T)
    if len(y_coords) > 0:
        ax.plot(delta_p_vals[x_coords], inv_delta2_vals[y_coords], 
                'o', markerfacecolor='none', markeredgecolor='limegreen',
                 markersize=8, markeredgewidth=1.5, label='Langevin-Hard')

    ax.set_xlabel(r'$\Delta_p$ (tensor noise)', fontsize=12)
    ax.set_ylabel(r'$(\Delta_2)^{-1}$ (matrix noise)', fontsize=12)
    ax.set_title(f'Empirical Phase Diagram (N={N}, p={p})', fontsize=14)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()