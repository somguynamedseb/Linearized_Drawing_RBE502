"""
Load a trajectory.json file and produce (qd_fun, qd_dot_fun, qd_ddot_fun)
for a computed-torque controller.

Two safeguards against insane joint values:
  1. After every IK call, wrap each joint to [-pi, pi].
  2. After all waypoints are solved, "unwrap" the sequence so adjacent
     waypoints are continuous (np.unwrap). This eliminates branch-flip
     spikes in the spline.

The trajectory the spline sees is monotonically continuous; the values
fed to the controller may exceed [-pi, pi] only by small offsets needed
to keep continuity, never by tens of radians.
"""

import json
import numpy as np
from scipy.interpolate import make_interp_spline


_DEFAULT_SEEDS = [
    ("zeros",                   np.zeros(6)),
    ("shoulder -90",            np.array([0, -np.pi/2, 0,        0, 0,        0])),
    ("shoulder -90, elbow +90", np.array([0, -np.pi/2, np.pi/2,  0, 0,        0])),
    ("wrist pitch +90",         np.array([0,  0,        0,       0, np.pi/2,  0])),
    ("shoulder -90, wrist +90", np.array([0, -np.pi/2,  0,       0, np.pi/2,  0])),
    ("elbow -90, wrist +90",    np.array([0,  0,       -np.pi/2, 0, np.pi/2,  0])),
]

IK_MAX_ITER  = 50
IK_TOL       = 1e-3
CLEAN_FK_TOL = 5e-3
N_RANDOM     = 30


def _wrap(q):
    """Wrap each joint to [-pi, pi]."""
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


def _solve_one(robot, x, y, z, R_ee, seed):
    T_sd = np.eye(4)
    T_sd[:3, :3] = R_ee
    T_sd[:3, 3]  = [x, y, z]
    q_sol, ok = robot.IK(T_sd, theta0=seed, tol=IK_TOL, max_iter=IK_MAX_ITER)
    q_sol = _wrap(q_sol)                                         # <-- key fix
    err = np.linalg.norm(robot.FK(q_sol)[:3, 3] - T_sd[:3, 3])
    return q_sol, ok, err


def _find_first_seed(robot, p0, R_ee, q_seed, verbose=True):
    candidates = []
    if q_seed is not None:
        candidates.append(("user-supplied", _wrap(q_seed)))
    candidates += list(_DEFAULT_SEEDS)

    if verbose:
        print(f"[trajectory] seed search for waypoint 0 "
              f"(target xyz = [{p0['x']:.3f}, {p0['y']:.3f}, {p0['z']:.3f}])")
        print(f"  {'seed':<28} {'ok':<6} {'fk_err [mm]':>12}")

    best = None
    for name, seed in candidates:
        q_sol, ok, err = _solve_one(robot, p0["x"], p0["y"], p0["z"], R_ee, seed)
        if verbose:
            print(f"  {name:<28} {str(ok):<6} {err*1000:>12.3f}")
        if ok and err < CLEAN_FK_TOL:
            if best is None or err < best[1]:
                best = (q_sol, err, name)

    if best is None:
        if verbose:
            print(f"[trajectory] curated seeds failed, trying {N_RANDOM} random seeds")
        rng = np.random.default_rng(42)
        for trial in range(N_RANDOM):
            seed = rng.uniform(-np.pi, np.pi, size=6)
            q_sol, ok, err = _solve_one(robot, p0["x"], p0["y"], p0["z"], R_ee, seed)
            if verbose:
                print(f"  random-{trial:02d}                  {str(ok):<6} {err*1000:>12.3f}")
            if ok and err < CLEAN_FK_TOL:
                best = (q_sol, err, f"random-{trial}")
                break

    if best is None:
        raise RuntimeError(
            "Could not solve IK at the first waypoint with any seed.\n"
            f"  target xyz = [{p0['x']:.4f}, {p0['y']:.4f}, {p0['z']:.4f}]\n"
            "Run debug_trajectory.py to investigate."
        )
    return best


def trajectory_from_json(path, robot, R_ee=None, q_seed=None, k=5, verbose=True):
    with open(path) as f:
        data = json.load(f)

    if R_ee is None:
        R_ee = np.eye(3)

    n = robot.jNum
    times = np.array([p["t"] for p in data], dtype=float)
    q_wp  = np.zeros((len(data), n))

    q0, err0, seed_name = _find_first_seed(robot, data[0], R_ee, q_seed, verbose)
    if verbose:
        print(f"[trajectory] using seed '{seed_name}' (FK err {err0*1e3:.3f} mm)")
    q_wp[0] = q0

    seed = q0.copy()
    failures = []
    for i in range(1, len(data)):
        p = data[i]
        q_sol, ok, err = _solve_one(robot, p["x"], p["y"], p["z"], R_ee, seed)
        q_wp[i] = q_sol
        if ok and err < CLEAN_FK_TOL:
            seed = q_sol
        else:
            failures.append(i)

    if failures and verbose:
        print(f"[trajectory] IK problems at {len(failures)}/{len(data)} waypoints "
              f"(first few: {failures[:5]})")

    # ---- Unwrap each joint along the time axis -----------------------------
    # Wrapping each waypoint individually to [-pi, pi] means two adjacent
    # waypoints can sit on opposite sides of the wrap, looking like a 2*pi
    # jump. np.unwrap removes those by adding/subtracting 2*pi.
    q_wp_unwrapped = np.unwrap(q_wp, axis=0)

    if verbose:
        added = q_wp_unwrapped - q_wp
        nonzero = np.any(np.abs(added) > 1e-9, axis=0)
        if nonzero.any():
            joints = [j+1 for j, flag in enumerate(nonzero) if flag]
            print(f"[trajectory] unwrap added 2*pi offsets on joints {joints}")

    # Drop duplicate timestamps
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, q_wp_unwrapped = times[keep], q_wp_unwrapped[keep]

    # Branch-flip warning AFTER unwrap — should now be quiet
    if len(q_wp_unwrapped) > 1:
        jumps = np.linalg.norm(np.diff(q_wp_unwrapped, axis=0), axis=1)
        if jumps.max() > 0.5 and verbose:
            idx = int(np.argmax(jumps))
            print(f"[trajectory] residual joint jump {jumps[idx]:.2f} rad at "
                  f"index {idx} — possible branch flip not fixable by unwrap.")

    if len(times) < k + 1:
        raise ValueError(f"Need at least {k+1} unique waypoints for spline order {k}.")

    if k == 5:
        bc = ([(1, np.zeros(n)), (2, np.zeros(n))],
              [(1, np.zeros(n)), (2, np.zeros(n))])
        spl = make_interp_spline(times, q_wp_unwrapped, k=5, bc_type=bc)
    elif k == 3:
        spl = make_interp_spline(times, q_wp_unwrapped, k=3, bc_type="clamped")
    else:
        spl = make_interp_spline(times, q_wp_unwrapped, k=k)

    spl_d   = spl.derivative(1)
    spl_dd  = spl.derivative(2)
    T_total = float(times[-1])
    qstart, qend = q_wp_unwrapped[0].copy(), q_wp_unwrapped[-1].copy()

    if verbose:
        print(f"[trajectory] joint range after processing: "
              f"[{q_wp_unwrapped.min():.3f}, {q_wp_unwrapped.max():.3f}] rad")

    def qd_fun(t):
        if t <= 0:        return qstart.copy()
        if t >= T_total:  return qend.copy()
        return np.asarray(spl(t))

    def qd_dot_fun(t):
        if t <= 0 or t >= T_total: return np.zeros(n)
        return np.asarray(spl_d(t))

    def qd_ddot_fun(t):
        if t <= 0 or t >= T_total: return np.zeros(n)
        return np.asarray(spl_dd(t))

    return qd_fun, qd_dot_fun, qd_ddot_fun, T_total