import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

import puma560 as puma
from trajectory_loader import trajectory_from_json


def clean_print(T, decimals=4):
    T_clean = np.where(np.abs(T) < 1e-10, 0.0, T)
    with np.printoptions(precision=decimals, suppress=True, floatmode='fixed'):
        print(T_clean)


# ============================================================================
# Robot, gains, trajectory
# ============================================================================
robot = puma.Puma560()
n     = robot.jNum

Kp = np.diag([100] * n)
Kv = np.diag([20]  * n)

R_down = np.diag([1.0, -1.0, -1.0])
qd_fun, qd_dot_fun, qd_ddot_fun, T_total = trajectory_from_json(
    "trajectory.json", robot, R_ee=R_down
)


# ============================================================================
# Computed-torque controller and closed-loop dynamics
# ============================================================================
def controller(q, qdot, t):
    e  = qd_fun(t)      - q
    ed = qd_dot_fun(t)  - qdot
    aq = qd_ddot_fun(t) + Kp @ e + Kv @ ed
    tau = (robot.mass_matrix(q) @ aq
           + robot.inverse_dynamics(q, qdot, np.zeros(n), g_vec=np.zeros(3))
           + robot.gravity_forces(q))
    return tau


def robot_ode(t, x):
    q, qdot = x[:n], x[n:2*n]
    tau = controller(q, qdot, t)
    M   = robot.mass_matrix(q)
    Cqd = robot.inverse_dynamics(q, qdot, np.zeros(n), g_vec=np.zeros(3))
    g   = robot.gravity_forces(q)
    qdd = np.linalg.solve(M, tau - Cqd - g)
    return np.concatenate([qdot, qdd])


# ============================================================================
# Simulate
# ============================================================================
T_sim   = T_total
dt      = 0.02
nsteps  = int(round(T_sim / dt)) + 1
t_eval  = np.linspace(0.0, T_sim, nsteps)
q_start = qd_fun(0.0)
x0      = np.concatenate([q_start, np.zeros(n)])

print(f"Simulating {T_sim:.2f} s ({nsteps} steps) ...")
sol = solve_ivp(robot_ode, [0, T_sim], x0,
                t_eval=t_eval, rtol=1e-6, atol=1e-8)
print("done.")

t_hist  = sol.t
q_hist  = sol.y[:n,    :]                 # (n, N)
qd_hist = sol.y[n:2*n, :]
N       = len(t_hist)


# ============================================================================
# FK on both target and actual to get Cartesian trajectories
# ============================================================================
target_q   = np.zeros((n, N))
target_xyz = np.zeros((3, N))
actual_xyz = np.zeros((3, N))
for i, t in enumerate(t_hist):
    target_q[:, i]   = qd_fun(t)
    target_xyz[:, i] = robot.FK(target_q[:, i])[:3, 3]
    actual_xyz[:, i] = robot.FK(q_hist[:, i])[:3, 3]

cart_err = np.linalg.norm(target_xyz - actual_xyz, axis=0)
print(f"Cartesian error: peak {cart_err.max()*1000:.2f} mm, "
      f"RMS {np.sqrt((cart_err**2).mean())*1000:.2f} mm")


# ============================================================================
# Save for the stick-model visualizer
# ============================================================================
np.savez("simulation.npz",
         t=t_hist,
         q=q_hist,             # (n, N) actual joint trajectory
         qdot=qd_hist,         # (n, N) actual joint velocity
         q_target=target_q,    # (n, N) desired joint trajectory
         target_xyz=target_xyz,
         actual_xyz=actual_xyz,
         dt=dt)
print(f"Saved simulation.npz  ({n} joints x {N} samples, dt={dt} s)")


# ============================================================================
# Plot 1 — joint tracking
# ============================================================================
fig1, axes1 = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
fig1.suptitle("Joint tracking — target vs actual")
for j in range(n):
    ax = axes1.flat[j]
    ax.plot(t_hist, target_q[j], 'k--', lw=1,   label='target')
    ax.plot(t_hist, q_hist[j],   'b',   lw=1.2, label='actual')
    ax.set_ylabel(f"q{j+1} [rad]")
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(loc='best', fontsize=8)
for ax in axes1[-1, :]:
    ax.set_xlabel("t [s]")
fig1.tight_layout()


# ============================================================================
# Plot 2 — Cartesian tracking
# ============================================================================
fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for k, lbl in enumerate(['x', 'y', 'z']):
    axes2[k].plot(t_hist, target_xyz[k], 'k--', lw=1,   label='target')
    axes2[k].plot(t_hist, actual_xyz[k], 'b',   lw=1.2, label='actual')
    axes2[k].set_ylabel(f"{lbl} [m]")
    axes2[k].grid(alpha=0.3)
axes2[0].legend(loc='best', fontsize=9)
axes2[-1].set_xlabel("t [s]")
fig2.suptitle("End-effector position — target vs actual")
fig2.tight_layout()


# ============================================================================
# Plot 3 — tracking error
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(t_hist, cart_err * 1000, 'r')
ax3.set_xlabel("t [s]")
ax3.set_ylabel("Cartesian error [mm]")
ax3.set_title(f"EE tracking error  "
              f"(peak {cart_err.max()*1000:.2f} mm, "
              f"RMS {np.sqrt((cart_err**2).mean())*1000:.2f} mm)")
ax3.grid(alpha=0.3)
fig3.tight_layout()


# ============================================================================
# Animation — top-down XY view
# ============================================================================
MIN_R, MAX_R = 0.2, 0.8

fig4, ax4 = plt.subplots(figsize=(7.5, 7.5))
ax4.set_aspect('equal')
ax4.set_xlim(-MAX_R*1.15, MAX_R*1.15)
ax4.set_ylim(-MAX_R*1.15, MAX_R*1.15)
ax4.set_xlabel("x [m]")
ax4.set_ylabel("y [m]")
ax4.set_title("Top-down view (XY plane)")
ax4.grid(alpha=0.3)

for r in (MIN_R, MAX_R):
    ax4.add_patch(mpatches.Circle((0, 0), r, fill=False, edgecolor='red', lw=1.5))
ax4.plot(0, 0, 'k+', ms=10)
ax4.plot(target_xyz[0], target_xyz[1], 'k--', lw=1, alpha=0.4, label='target path')

actual_line, = ax4.plot([], [], 'b-', lw=1.6, label='actual path')
target_dot,  = ax4.plot([], [], 'ko', ms=7, label='target')
actual_dot,  = ax4.plot([], [], 'bo', ms=7, label='EE')
time_txt = ax4.text(0.02, 0.97, '', transform=ax4.transAxes, va='top',
                    fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=0.85))
ax4.legend(loc='lower right', fontsize=9)


def init():
    actual_line.set_data([], [])
    target_dot.set_data([], [])
    actual_dot.set_data([], [])
    time_txt.set_text('')
    return actual_line, target_dot, actual_dot, time_txt


def update(i):
    actual_line.set_data(actual_xyz[0, :i+1], actual_xyz[1, :i+1])
    target_dot.set_data([target_xyz[0, i]], [target_xyz[1, i]])
    actual_dot.set_data([actual_xyz[0, i]], [actual_xyz[1, i]])
    time_txt.set_text(f"t = {t_hist[i]:.2f} s")
    return actual_line, target_dot, actual_dot, time_txt


ani = FuncAnimation(fig4, update, frames=N, init_func=init,
                    interval=int(dt * 1000), blit=True, repeat=False)

plt.show()