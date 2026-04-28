"""
Compute and display the reachable workspace (slice at constant z) for the Puma560.

Approach: Monte-Carlo sample joint configurations, run FK, keep points within a
thin slab around the target height, then outline the region with a convex hull.

Outer hull vertex furthest from the base axis = the robot's max reach at that height.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from puma560 import Puma560


def reach_envelope(robot, z_target, z_tol=0.02, n_samples=50000, joint_limits=None):
    """
    Sample the reachable workspace at height z = z_target (+/- z_tol).

    Parameters
    ----------
    robot        : Puma560 instance
    z_target     : target end-effector height [m]
    z_tol        : slab half-thickness [m] — tighter = cleaner slice but fewer points
    n_samples    : number of random joint configurations to try
    joint_limits : list of (low, high) per joint. Defaults to +/- pi for all joints.

    Returns
    -------
    points (N, 2) : (x, y) positions of end-effector inside the slab
    """
    if joint_limits is None:
        joint_limits = [(-np.pi, np.pi)] * robot.jNum

    lows  = np.array([lo for lo, hi in joint_limits])
    highs = np.array([hi for lo, hi in joint_limits])

    pts = []
    for _ in range(n_samples):
        theta = np.random.uniform(lows, highs)
        T = robot.FK(theta)
        if abs(T[2, 3] - z_target) < z_tol:
            pts.append(T[0:2, 3])
    return np.array(pts) if pts else np.empty((0, 2))


def plot_reach(points, z_target, robot=None, ax=None):
    """Plot reachable points and the convex-hull envelope."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if len(points) < 3:
        ax.set_title(f'Not enough points at z = {z_target:.3f} m (got {len(points)})')
        return ax

    # Scatter the reachable points
    ax.scatter(points[:, 0], points[:, 1], s=4, color='steelblue', alpha=0.3,
               label=f'Reachable points (N={len(points)})')

    # Outline with convex hull
    hull = ConvexHull(points)
    hull_pts = np.vstack([points[hull.vertices], points[hull.vertices[0]]])  # close the loop
    ax.plot(hull_pts[:, 0], hull_pts[:, 1], 'r-', linewidth=2, label='Envelope (convex hull)')

    # Max-reach circle for reference
    dists = np.linalg.norm(points, axis=1)
    max_reach = dists.max()
    phi = np.linspace(0, 2*np.pi, 100)
    ax.plot(max_reach*np.cos(phi), max_reach*np.sin(phi), 'g--',
            linewidth=1.2, label=f'Max reach = {max_reach:.3f} m')

    # Base pedestal footprint (optional, for context)
    pedestal_r = 0.15
    ax.plot(pedestal_r*np.cos(phi), pedestal_r*np.sin(phi), 'k-',
            linewidth=1.5, label='Base pedestal')

    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(f'Reachable workspace slice  (z = {z_target:.3f} m)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Print numeric summary
    print(f"z = {z_target:.3f} m")
    print(f"  Reachable points sampled : {len(points)}")
    print(f"  Max reach  (from z-axis) : {max_reach:.4f} m")
    print(f"  Min reach  (from z-axis) : {dists.min():.4f} m")
    print(f"  X range                  : [{points[:,0].min():+.3f}, {points[:,0].max():+.3f}] m")
    print(f"  Y range                  : [{points[:,1].min():+.3f}, {points[:,1].max():+.3f}] m")
    print(f"  Envelope hull vertices   : {len(hull.vertices)}")

    return ax


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    np.random.seed(0)
    robot = Puma560()

    # --- Pick height(s) to analyze ---
    Z_TARGET = 0.2        # change this to slice at a different height
    Z_TOL    = 0.02       # +/- slab thickness (smaller = sharper, need more samples)
    N        = 100000     # sample count

    # --- Optional: realistic joint limits instead of full +/- pi ---
    # Comment this out to use full rotation.
    joint_limits = [
        (-2.79,   2.79),  
        (-3.93,   0.79),  
        (-0.79,   3.93),  
        (-1.92,   2.97),  
        (-1.75,   1.75),  
        (-4.64,   4.64),  
    ]
    
    # pts = reach_envelope(robot, z_target=Z_TARGET, z_tol=Z_TOL, n_samples=N)
    pts = reach_envelope(robot, z_target=Z_TARGET, z_tol=Z_TOL, n_samples=N, joint_limits=joint_limits)
    plot_reach(pts, z_target=Z_TARGET, robot=robot)
    plt.tight_layout()
    plt.show()
