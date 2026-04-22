import numpy as np
import puma560 as puma
from scipy.integrate import solve_ivp

def clean_print(T, decimals=4):
    T_clean = np.where(np.abs(T) < 1e-10, 0.0, T)
    with np.printoptions(precision=decimals, suppress=True, floatmode='fixed'):
        print(T_clean)
        
robot = puma.Puma560()

# === Desired trajectory (user-defined) ===
# Each returns a 6-vector at time t
def qd_fun(t):       ...   # desired joint positions
def qd_dot_fun(t):   ...   # desired joint velocities
def qd_ddot_fun(t):  ...   # desired joint accelerations

# === Gains ===
n = robot.jNum
Kp = np.diag([100]*n)   # tune these
Kv = np.diag([20]*n)    # critical damping: Kv ≈ 2*sqrt(Kp)

# === Computed-torque controller ===
def controller(q, qdot, t):
    e   = qd_fun(t)      - q
    ed  = qd_dot_fun(t)  - qdot
    aq  = qd_ddot_fun(t) + Kp @ e + Kv @ ed   # "virtual" acceleration command

    tau = robot.mass_matrix(q) @ aq \
        + robot.inverse_dynamics(q, qdot, np.zeros(n), g_vec=np.zeros(3)) \
        + robot.gravity_forces(q)
    return tau

# === Closed-loop dynamics for the integrator ===
def robot_ode(t, x):
    q    = x[0:n]
    qdot = x[n:2*n]
    tau  = controller(q, qdot, t)

    M    = robot.mass_matrix(q)
    Cqd  = robot.inverse_dynamics(q, qdot, np.zeros(n), g_vec=np.zeros(3))  # C q̇
    g    = robot.gravity_forces(q)
    qdd  = np.linalg.solve(M, tau - Cqd - g)
    return np.concatenate([qdot, qdd])

# === Simulate ===
T_sim = 5.0
t_eval = np.arange(0, T_sim + 0.01, 0.01)
q_start = np.zeros(n)
x0 = np.concatenate([q_start, np.zeros(n)])     # [q; qdot]

sol = solve_ivp(robot_ode, [0, T_sim], x0, t_eval=t_eval, rtol=1e-6, atol=1e-8)

t_hist = sol.t
q_hist = sol.y[0:n, :]      # shape (n, N)
qd_hist = sol.y[n:2*n, :]
    
    
    
    
        