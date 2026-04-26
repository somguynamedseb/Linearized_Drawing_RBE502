"""
Load a trajectory.json file produced by trajectory_drawer.py and turn it into
(qd_fun, qd_dot_fun, qd_ddot_fun) for a computed-torque controller.

Robust seeding strategy:
  - try a curated list of seed candidates at waypoint 0
  - if none clears the FK round-trip threshold, try random seeds
  - print a diagnostic table so you can see why each seed succeeded or failed
  - guard the warm-start: only update seed on clean solutions
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
CLEAN_FK_TOL = 5e-3   # 5 mm; we'll FK-verify the spline anyway
N_RANDOM     = 30     # random-seed retries if curated list comes up empty


def _solve_one(robot, x, y, z, R_ee, seed):
    T_sd = np.eye(4)
    T_sd[:3, :3] = R_ee
    T_sd[:3, 3]  = [x, y, z]
    q_sol, ok = robot.IK(T_sd, theta0=seed, tol=IK_TOL, max_iter=IK_MAX_ITER)
    err = np.linalg.norm(robot.FK(q_sol)[:3, 3] - T_sd[:3, 3])
    return q_sol, ok, err


def _find_first_seed(robot, p0, R_ee, q_seed, verbose=True):
    candidates = []
    if q_seed is not None:
        candidates.append(("user-supplied", np.asarray(q_seed, dtype=float)))
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

    # Fallback — random seeds in [-pi, pi]^6
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
            f"  R_ee =\n{R_ee}\n"
            "Run debug_trajectory.py to investigate. The pose may be unreachable "
            "with this orientation, or the IK tolerance may need adjusting."
        )
    return best


def trajectory_from_json(path, robot, R_ee=None, q_seed=None, k=5, verbose=True):
    """
    Parameters
    ----------
    path : str
    robot : Puma560
    R_ee : (3,3) ndarray, optional
        End-effector orientation. Default = identity. For drawing at Z<0 use
        np.diag([1, -1, -1]).
    q_seed : (6,) ndarray, optional
        Preferred seed for IK at waypoint 0. Tried before the default list.
    k : int
        Spline order; 5 = quintic (recommended for computed torque).
    verbose : bool
        Print the seed-search table.

    Returns
    -------
    qd_fun, qd_dot_fun, qd_ddot_fun : callable(t) -> (n,) ndarray
    T_total : float
    """
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

    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, q_wp = times[keep], q_wp[keep]

    if len(q_wp) > 1:
        jumps = np.linalg.norm(np.diff(q_wp, axis=0), axis=1)
        if jumps.max() > 0.5 and verbose:
            idx = int(np.argmax(jumps))
            print(f"[trajectory] large joint jump {jumps[idx]:.2f} rad at index {idx} "
                  f"— possible branch flip.")

    if len(times) < k + 1:
        raise ValueError(f"Need at least {k+1} unique waypoints for spline order {k}.")

    if k == 5:
        bc = ([(1, np.zeros(n)), (2, np.zeros(n))],
              [(1, np.zeros(n)), (2, np.zeros(n))])
        spl = make_interp_spline(times, q_wp, k=5, bc_type=bc)
    elif k == 3:
        spl = make_interp_spline(times, q_wp, k=3, bc_type="clamped")
    else:
        spl = make_interp_spline(times, q_wp, k=k)

    spl_d   = spl.derivative(1)
    spl_dd  = spl.derivative(2)
    T_total = float(times[-1])
    qstart, qend = q_wp[0].copy(), q_wp[-1].copy()

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