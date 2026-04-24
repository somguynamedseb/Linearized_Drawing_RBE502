import numpy as np
from scipy.integrate import solve_ivp

from puma560 import Puma560
from puma_visualizer import PumaVisualizer

# ============================================================
# Setup
# ============================================================
robot = Puma560()
viz = PumaVisualizer(robot)


# ============================================================
# Example 1 — Static pose
# ============================================================
viz.show_pose(np.array([np.pi/4, -np.pi/6, np.pi/3, 0, np.pi/4, 0]),
              title='Test pose')


# ============================================================
# Example 2 — Animate a hand-made trajectory
# Simple sinusoidal sweep on each joint
# ============================================================
def joint_angles_at(t):
    return np.array([
        0.8 * np.sin(0.5 * t),
        0.4 * np.sin(0.7 * t) - 0.3,
        0.5 * np.sin(0.6 * t) + 0.5,
        0.6 * np.sin(0.9 * t),
        0.4 * np.sin(0.8 * t),
        0.5 * np.sin(1.0 * t),
    ])

t_eval = np.arange(0, 5, 0.02)
theta_traj = np.array([joint_angles_at(t) for t in t_eval])   # (N, 6)
viz.animate(theta_traj, dt=0.02, title='Sinusoidal trajectory')


# ============================================================
# Example 3 — Animate output of a simulation (passive, no control)
# Arm released from a tilted pose under gravity
# ============================================================
def robot_ode(t, x):
    n = robot.jNum
    q    = x[0:n]
    qdot = x[n:2*n]

    tau = np.zeros(n)   # no control torques — free fall under gravity

    M = robot.mass_matrix(q)
    # C q̇ computed as a vector in one Newton-Euler pass (fast)
    Cqd = robot.inverse_dynamics(q, qdot, np.zeros(n), g_vec=np.zeros(3))
    g   = robot.gravity_forces(q)

    qdd = np.linalg.solve(M, tau - Cqd - g)
    return np.concatenate([qdot, qdd])

T_sim = 3.0
t_eval = np.arange(0, T_sim + 0.02, 0.02)
q_start = np.array([0.0, -0.4, 0.8, 0.0, 0.3, 0.0])
x0 = np.concatenate([q_start, np.zeros(robot.jNum)])

sol = solve_ivp(robot_ode, [0, T_sim], x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)

theta_traj = sol.y[0:robot.jNum, :].T        # (N, 6)
viz.animate(theta_traj, dt=0.02, title='Free-fall simulation')


# ============================================================
# Example 4 — Save an animation to file
# Uncomment to save. Requires pillow for GIF, ffmpeg for MP4.
# ============================================================
# viz.animate(theta_traj, dt=0.02, save_as='puma_demo.gif')