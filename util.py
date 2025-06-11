import numpy as np
import xarray as xr
import re
import os
import torch


def load_data(output_dir, var="correlation"):
    if var not in {"correlation", "signal", "trajectory"}:
        raise ValueError(f"{var} is not a valid saved variable")

    # Match filenames like: langevin_N{N}_D2{delta2}_Dp{deltap}_trial{trial}.pt
    pattern = re.compile(r"langevin_N(\d+)_D2([0-9.]+)_Dp([0-9.]+)_trial(\d+).pt")
    results = []

    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if not match:
            continue

        N, delta2, deltap, trial = match.groups()
        N = int(N)
        trial = int(trial)
        delta2 = float(delta2)
        deltap = float(deltap)

        data = torch.load(os.path.join(output_dir, fname))[var].numpy()
        results.append((trial, delta2, deltap, N, data))

    if not results:
        raise ValueError(f"No valid files found in {output_dir} for variable '{var}'")

    trials = sorted({r[0] for r in results})
    delta2s = sorted({r[1] for r in results})
    deltap_vals = sorted({r[2] for r in results})
    steps = results[0][4].shape[0]
    N_val = results[0][3]

    metadata = {
        "correlation": {
            "shape": (len(trials), len(delta2s), len(deltap_vals), steps),
            "dims": ["trial", "delta2", "deltap", "step"],
            "coords": {"trial": trials, "delta2": delta2s, "deltap": deltap_vals, "step": list(range(steps))}
        },
        "trajectory": {
            "shape": (len(trials), len(delta2s), len(deltap_vals), steps, N_val),
            "dims": ["trial", "delta2", "deltap", "step", "i"],
            "coords": {"trial": trials, "delta2": delta2s, "deltap": deltap_vals,
                       "step": list(range(steps)), "i": list(range(N_val))}
        },
        "signal": {
            "shape": (len(trials), len(delta2s), len(deltap_vals)),
            "dims": ["trial", "delta2", "deltap"],
            "coords": {"trial": trials, "delta2": delta2s, "deltap": deltap_vals}
        }
    }

    meta = metadata[var]
    data_array = xr.DataArray(
        data=np.full(meta["shape"], np.nan),
        dims=meta["dims"],
        coords=meta["coords"]
    )

    for trial, delta2, deltap, _, data in results:
        indexer = dict(trial=trial, delta2=delta2, deltap=deltap)
        data_array.loc[indexer] = data

    return data_array

