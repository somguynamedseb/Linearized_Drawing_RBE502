"""
Load simulation.npz produced by main.py and animate the joint trajectory
as a 3D stick figure using PumaVisualizer.

Run main.py first to generate simulation.npz, then run this script.
"""

import numpy as np
import puma560 as puma
from puma_visualizer import PumaVisualizer


# Choose what to animate: "actual" (simulated arm), "target" (commanded
# trajectory), or "both" (run them back-to-back in separate windows).
MODE = "actual"

# Frame stride — set >1 to speed up playback for long trajectories
# (e.g. STRIDE = 2 plays at 2x real time).
STRIDE = 1


def main():
    data = np.load("simulation.npz")
    t        = data["t"]
    q_hist   = data["q"]            # (n, N)
    q_target = data["q_target"]     # (n, N)
    dt       = float(data["dt"])

    n, N = q_hist.shape
    print(f"Loaded {N} samples, {n} joints, dt = {dt} s, "
          f"duration = {t[-1]:.2f} s")

    robot = puma.Puma560()
    viz   = PumaVisualizer(robot)

    # PumaVisualizer.animate expects shape (N, n)
    if MODE in ("actual", "both"):
        traj = q_hist.T[::STRIDE]
        viz.animate(traj, dt=dt * STRIDE,
                    title="Simulated arm — actual trajectory")

    if MODE in ("target", "both"):
        traj = q_target.T[::STRIDE]
        viz.animate(traj, dt=dt * STRIDE,
                    title="Commanded target trajectory")


if __name__ == "__main__":
    main()
