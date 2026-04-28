"""
ik_debug.py — Sanity checks for the Puma560 IK.

Runs four tests:
  1. Round-trip: FK(q) -> IK should recover a pose that re-FKs to the same T.
  2. Limit handling: targets at and beyond joint limits.
  3. Singularity: wrist-aligned configurations (q5 = 0).
  4. Workspace sweep: random reachable poses, report success rate.

Each test prints PASS/FAIL plus the residuals so you can see where it breaks.
"""

import numpy as np
import puma560 as puma


# ---------- helpers ----------------------------------------------------------

def pose_err(T_a, T_b):
    """Returns (position error in meters, rotation error in radians)."""
    p_err = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])
    R_err = T_a[:3, :3] @ T_b[:3, :3].T
    cos_th = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    return p_err, np.arccos(cos_th)


def in_limits(robot, q, slack=1e-6):
    return bool(np.all(q >= robot.q_min - slack) and np.all(q <= robot.q_max + slack))


def fmt_q(q):
    return "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"


def banner(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


# ---------- the tests --------------------------------------------------------

def test_roundtrip(robot, n_cases=20, seed=0):
    """FK on random q, then IK back. Should converge with tiny residual."""
    banner("TEST 1 — Round-trip FK -> IK")
    rng = np.random.default_rng(seed)
    fails = 0
    worst_p, worst_r = 0.0, 0.0

    for k in range(n_cases):
        # Sample inside limits, with a small margin
        margin = 0.1
        q_true = rng.uniform(robot.q_min + margin, robot.q_max - margin)
        T_target = robot.FK(q_true)

        # Start from home so we don't cheat by seeding near the answer
        q_sol, ok = robot.IK(T_target, theta0=np.zeros(robot.jNum))
        T_sol = robot.FK(q_sol)
        ep, er = pose_err(T_target, T_sol)
        worst_p = max(worst_p, ep)
        worst_r = max(worst_r, er)

        bad = (not ok) or ep > 1e-3 or er > 1e-3 or not in_limits(robot, q_sol)
        if bad:
            fails += 1
            print(f"  case {k:2d}: ok={ok}  pos_err={ep*1000:7.3f} mm  "
                  f"rot_err={np.degrees(er):6.3f} deg  "
                  f"in_lim={in_limits(robot, q_sol)}")
            print(f"            q_true = {fmt_q(q_true)}")
            print(f"            q_sol  = {fmt_q(q_sol)}")

    print(f"\n  {n_cases - fails}/{n_cases} passed.  "
          f"Worst pos {worst_p*1000:.3f} mm, worst rot {np.degrees(worst_r):.3f} deg.")


def test_limit_handling(robot):
    """Target a pose just outside limits — IK should clamp and report failure cleanly."""
    banner("TEST 2 — Limit handling")

    # Build a target by FK'ing a configuration that violates joint 2 by a lot
    q_bad = np.zeros(robot.jNum)
    q_bad[1] = robot.q_min[1] - 1.0   # 1 rad past the lower stop on joint 2
    T_target = robot.FK(q_bad)        # FK happily ignores limits, so this works

    q_sol, ok = robot.IK(T_target, theta0=np.zeros(robot.jNum))
    print(f"  unreachable target: ok={ok} (expected False)")
    print(f"  q_sol           = {fmt_q(q_sol)}")
    print(f"  in_limits       = {in_limits(robot, q_sol)} (expected True)")
    ep, er = pose_err(T_target, robot.FK(q_sol))
    print(f"  residual pos    = {ep*1000:.2f} mm   rot = {np.degrees(er):.2f} deg")
    print("  (residual should be > 0 — solver got as close as joint limits allow)")

    # And a reachable target: should pass
    q_ok = np.array([0.5, -1.0, 1.5, 0.0, 0.3, 0.0])
    T_ok = robot.FK(q_ok)
    q_sol2, ok2 = robot.IK(T_ok, theta0=np.zeros(robot.jNum))
    ep2, er2 = pose_err(T_ok, robot.FK(q_sol2))
    print(f"\n  reachable target:  ok={ok2}  in_lim={in_limits(robot, q_sol2)}  "
          f"pos_err={ep2*1000:.3f} mm  rot_err={np.degrees(er2):.3f} deg")


def test_singularity(robot):
    """Wrist singularity is at q5 = 0 (joints 4 and 6 axes align)."""
    banner("TEST 3 — Wrist singularity (q5 ≈ 0)")
    for q5 in [0.0, 1e-3, 1e-2, 0.1]:
        q_true = np.array([0.3, -1.0, 1.5, 0.4, q5, -0.2])
        T_target = robot.FK(q_true)
        q_sol, ok = robot.IK(T_target, theta0=np.zeros(robot.jNum))
        ep, er = pose_err(T_target, robot.FK(q_sol))
        # Check Jacobian conditioning at the truth
        Js = robot._space_jacobian(q_true)
        cond = np.linalg.cond(Js)
        print(f"  q5={q5:+.3f}  ok={ok}  pos_err={ep*1000:6.2f} mm  "
              f"rot_err={np.degrees(er):6.2f} deg  cond(J)={cond:.1e}")
    print("  (cond(J) blows up as q5 -> 0; damped LS should still converge)")


def test_workspace_sweep(robot, n_cases=200, seed=1):
    """Random reachable targets. Report success rate and worst residuals."""
    banner(f"TEST 4 — Workspace sweep ({n_cases} random reachable targets)")
    rng = np.random.default_rng(seed)
    successes = 0
    pos_errs, rot_errs = [], []

    for _ in range(n_cases):
        q_true = rng.uniform(robot.q_min + 0.05, robot.q_max - 0.05)
        T_target = robot.FK(q_true)
        q_sol, ok = robot.IK(T_target, theta0=np.zeros(robot.jNum))
        ep, er = pose_err(T_target, robot.FK(q_sol))
        pos_errs.append(ep)
        rot_errs.append(er)
        if ok and ep < 1e-3 and er < 1e-3 and in_limits(robot, q_sol):
            successes += 1

    pe = np.array(pos_errs) * 1000
    re = np.degrees(np.array(rot_errs))
    print(f"  success rate: {successes}/{n_cases} ({100*successes/n_cases:.1f}%)")
    print(f"  pos error  — median {np.median(pe):.3f} mm   p95 {np.percentile(pe, 95):.3f} mm   max {pe.max():.3f} mm")
    print(f"  rot error  — median {np.median(re):.3f} deg  p95 {np.percentile(re, 95):.3f} deg  max {re.max():.3f} deg")
    if successes / n_cases < 0.95:
        print("  --> low success rate suggests damping (lam) too high, max_iter too low,")
        print("      or singularities/limits being hit more often than expected.")


# ---------- main -------------------------------------------------------------

if __name__ == "__main__":
    robot = puma.Puma560()
    print("Puma560 IK debug")
    print(f"  q_min = {fmt_q(robot.q_min)}")
    print(f"  q_max = {fmt_q(robot.q_max)}")
    print(f"  M (home EE position) = {fmt_q(robot.M[:3, 3])}")

    test_roundtrip(robot)
    test_limit_handling(robot)
    test_singularity(robot)
    test_workspace_sweep(robot)