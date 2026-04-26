"""
Debug the trajectory.json -> joint trajectory pipeline.

Stages:
  1. Load JSON and plot the raw drawn (x,y) path with the workspace circles.
  2. Try a battery of IK seeds at waypoint 0. Report convergence and the
     FK round-trip error for each (i.e. how well IK actually solved).
  3. Using the best seed, run IK across the whole path with warm-starting.
     Plot the per-waypoint joint values and per-waypoint FK error.
  4. Fit the quintic spline and plot q(t), qdot(t), qddot(t).
  5. FK the spline and overlay it on the original drawing as a sanity check.

Run this on its own — it does NOT use trajectory_loader.py, so we can see
each step in isolation.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

import puma560 as puma

# ============================================================================
# Config
# ============================================================================
JSON_PATH = "trajectory.json"

# Try this orientation first. If everything below fails, try R_EE = np.eye(3)
# to confirm the position part is reachable, then re-introduce orientation.
R_EE = np.diag([1.0, -1.0, -1.0])   # tool axis -> -world z

# Seed candidates for the FIRST waypoint. Add more if none converge.
SEED_CANDIDATES = [
    ("zeros",                   np.zeros(6)),
    ("shoulder -90",            np.array([0, -np.pi/2, 0,        0, 0,        0])),
    ("shoulder -90, elbow +90", np.array([0, -np.pi/2, np.pi/2,  0, 0,        0])),
    ("wrist pitch +90",         np.array([0,  0,        0,       0, np.pi/2,  0])),
    ("shoulder -90, wrist +90", np.array([0, -np.pi/2,  0,       0, np.pi/2,  0])),
    ("elbow -90, wrist +90",    np.array([0,  0,       -np.pi/2, 0, np.pi/2,  0])),
]

IK_MAX_ITER = 50    # bumped from the default 20 to give convergence more room
IK_TOL      = 1e-3


# ============================================================================
# 1. Load and view the raw drawing
# ============================================================================
with open(JSON_PATH) as f:
    data = json.load(f)

t_wp = np.array([p["t"] for p in data])
x_wp = np.array([p["x"] for p in data])
y_wp = np.array([p["y"] for p in data])
z_wp = np.array([p["z"] for p in data])
N    = len(data)
print(f"Loaded {N} waypoints, T_total = {t_wp[-1]:.3f} s, "
      f"z = {z_wp[0]:.3f} m (constant)")

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')
for r in (0.2, 0.8):
    ax.add_patch(mpatches.Circle((0, 0), r, fill=False, edgecolor='red', lw=1.5))
ax.plot(x_wp, y_wp, 'b-', lw=1)
ax.plot(x_wp[0],  y_wp[0],  'go', ms=10, label='start')
ax.plot(x_wp[-1], y_wp[-1], 'rs', ms=10, label='end')
ax.set_title(f"Drawn trajectory  ({N} pts, z = {z_wp[0]:.2f} m)")
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.grid(alpha=0.3); ax.legend()


# ============================================================================
# 2. Try seeds at waypoint 0
# ============================================================================
robot = puma.Puma560()
T0 = np.eye(4); T0[:3, :3] = R_EE; T0[:3, 3] = [x_wp[0], y_wp[0], z_wp[0]]

print(f"\n--- Testing IK seeds at waypoint 0  (target xyz = "
      f"{T0[:3, 3].round(3).tolist()}) ---")
print(f"{'seed':<28} {'ok':<6} {'FK err [mm]':>12}  {'q solution':<}")

best = None
for name, seed in SEED_CANDIDATES:
    q_sol, ok = robot.IK(T0, theta0=seed, tol=IK_TOL, max_iter=IK_MAX_ITER)
    err = np.linalg.norm(robot.FK(q_sol)[:3, 3] - T0[:3, 3])
    print(f"{name:<28} {str(ok):<6} {err*1000:>12.3f}  "
          f"{np.round(q_sol, 2).tolist()}")
    # Treat as 'good' if FK error is small AND the solution is in a sane range
    if ok and err < 1e-3 and np.max(np.abs(q_sol)) < 2 * np.pi:
        if best is None or err < best[1]:
            best = (q_sol, err, name)

if best is None:
    print("\n*** No seed produced a clean solution at waypoint 0. ***")
    print("    Suggestions:")
    print("    - Try R_EE = np.eye(3) to check the position is reachable at all")
    print("    - Add more seed candidates that better match your tool convention")
    print("    - Verify z = -0.24 is reachable for your Puma (the home EE is at z ≈ +0.86)")
    plt.show()
    raise SystemExit(1)

q_seed_use, err0, seed_name = best
print(f"\nBest seed: '{seed_name}'  (FK err {err0*1e3:.3f} mm)\n")


# ============================================================================
# 3. Full IK with warm-start
# ============================================================================
q_wp     = np.zeros((N, 6))
ok_flags = np.zeros(N, dtype=bool)
fk_err   = np.zeros(N)

seed = q_seed_use.copy()
for i, p in enumerate(data):
    Ti = np.eye(4); Ti[:3, :3] = R_EE; Ti[:3, 3] = [p["x"], p["y"], p["z"]]
    q_sol, ok = robot.IK(Ti, theta0=seed, tol=IK_TOL, max_iter=IK_MAX_ITER)
    q_wp[i]     = q_sol
    ok_flags[i] = ok
    fk_err[i]   = np.linalg.norm(robot.FK(q_sol)[:3, 3] - Ti[:3, 3])
    if ok and fk_err[i] < 1e-3:
        seed = q_sol            # only warm-start from clean solutions

print(f"IK summary:")
print(f"  converged:       {ok_flags.sum()}/{N}")
print(f"  FK error  max:   {fk_err.max()*1000:.3f} mm")
print(f"  FK error  mean:  {fk_err.mean()*1000:.3f} mm")
print(f"  joint range:     [{q_wp.min():.2f}, {q_wp.max():.2f}] rad")

# Per-waypoint joint plot — branch flips show as vertical jumps here
fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
fig.suptitle("Joint solutions per waypoint (look for jumps = branch flips)")
for j in range(6):
    ax = axes.flat[j]
    ax.plot(t_wp, q_wp[:, j], 'b.-', ms=3, lw=0.8)
    bad = ~ok_flags
    if bad.any():
        ax.plot(t_wp[bad], q_wp[bad, j], 'rx', ms=8, label='IK failed')
        ax.legend(fontsize=8)
    ax.set_ylabel(f"q{j+1} [rad]"); ax.grid(alpha=0.3)
for ax in axes[-1, :]:
    ax.set_xlabel("t [s]")
fig.tight_layout()

# Per-waypoint FK round-trip error
fig, ax = plt.subplots(figsize=(10, 3))
ax.semilogy(t_wp, np.maximum(fk_err, 1e-9) * 1000, 'r.-', ms=4)
ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='1 mm')
ax.set_xlabel("t [s]"); ax.set_ylabel("FK round-trip error [mm]")
ax.set_title("Per-waypoint IK accuracy")
ax.grid(alpha=0.3, which='both'); ax.legend()
fig.tight_layout()


# ============================================================================
# 4. Fit the spline and plot derivatives
# ============================================================================
keep = np.concatenate([[True], np.diff(t_wp) > 1e-9])
ts, qs = t_wp[keep], q_wp[keep]

bc = ([(1, np.zeros(6)), (2, np.zeros(6))],
      [(1, np.zeros(6)), (2, np.zeros(6))])
spl    = make_interp_spline(ts, qs, k=5, bc_type=bc)
spl_d  = spl.derivative(1)
spl_dd = spl.derivative(2)

t_fine = np.linspace(ts[0], ts[-1], 1000)
q_f, qd_f, qdd_f = spl(t_fine), spl_d(t_fine), spl_dd(t_fine)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle("Fitted spline trajectory")
for j in range(6):
    axes[0].plot(t_fine, q_f[:,   j], label=f"q{j+1}")
    axes[1].plot(t_fine, qd_f[:,  j])
    axes[2].plot(t_fine, qdd_f[:, j])
axes[0].set_ylabel("q [rad]");      axes[0].legend(fontsize=8, ncol=6)
axes[1].set_ylabel("q̇ [rad/s]")
axes[2].set_ylabel("q̈ [rad/s²]");  axes[2].set_xlabel("t [s]")
for a in axes: a.grid(alpha=0.3)
fig.tight_layout()

print(f"\nSpline derivative magnitudes:")
print(f"  max |q̇|   = {np.abs(qd_f).max():.2f} rad/s")
print(f"  max |q̈|   = {np.abs(qdd_f).max():.2f} rad/s²")


# ============================================================================
# 5. FK the spline back to Cartesian — this is the smoking-gun plot
# ============================================================================
xyz_fine = np.array([robot.FK(q_f[i])[:3, 3] for i in range(len(t_fine))])

fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.set_aspect('equal')
for r in (0.2, 0.8):
    ax.add_patch(mpatches.Circle((0, 0), r, fill=False, edgecolor='red', lw=1.5))
ax.plot(x_wp,            y_wp,            'k.', ms=4, alpha=0.5, label='waypoints')
ax.plot(xyz_fine[:, 0],  xyz_fine[:, 1],  'b-', lw=1.5,           label='FK(spline)')
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.set_title("Spline through joint space, FK'd back to xy")
ax.grid(alpha=0.3); ax.legend()

# z drift check — should stay at the constant drawing height
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(t_fine, xyz_fine[:, 2], 'b', label='FK(spline) z')
ax.axhline(z_wp[0], color='k', ls='--', lw=0.8, label=f'target z = {z_wp[0]:.3f}')
ax.set_xlabel("t [s]"); ax.set_ylabel("z [m]")
ax.set_title("Z drift along the spline (should be flat)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()

plt.show()
