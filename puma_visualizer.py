import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

class PumaVisualizer:
    """
    Stick-figure visualizer for a 6-DOF arm described by PoE (w, slist, M).
    Usage:
        viz = PumaVisualizer(robot)
        viz.show_pose(theta)                  # static pose
        viz.animate(theta_trajectory, dt=0.02) # animate a trajectory
    """

    def __init__(self, robot, Q=None):
        """
        robot : object with attributes .w (n,3), .slist (6,n), .M (4,4), .jNum
        Q     : (n,3) home positions of each joint in space frame.
                If None, tries robot.Q.
        """
        self.robot = robot
        self.n = robot.jNum
        self.Q = Q if Q is not None else robot.Q

    # ---------- Kinematics for link positions ----------
    def _exp6(self, i, theta_i):
        w = self.robot.w[i]
        v = self.robot.slist[3:6, i]
        wh = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        T = np.eye(4)
        T[0:3, 0:3] = np.eye(3) + np.sin(theta_i)*wh + (1 - np.cos(theta_i))*wh @ wh
        G = np.eye(3)*theta_i + (1 - np.cos(theta_i))*wh + (theta_i - np.sin(theta_i))*wh @ wh
        T[0:3, 3] = G @ v
        return T

    def joint_positions(self, theta):
        """Return (n+2, 3) array: base, joint 1 origin, ..., joint n origin, end-effector."""
        pts = [np.zeros(3)]                    # base at origin
        T = np.eye(4)
        for i in range(self.n):
            # Position of joint i's frame before joint i moves = cumulative * Q[i]
            p_joint = (T @ np.append(self.Q[i], 1.0))[0:3]
            pts.append(p_joint)
            T = T @ self._exp6(i, theta[i])    # apply joint i
        # End-effector
        p_ee = (T @ self.robot.M @ np.array([0, 0, 0, 1]))[0:3]
        pts.append(p_ee)
        return np.array(pts)

    # ---------- Plotting ----------
    def _setup_axes(self, ax):
        reach = 1.2
        ax.set_xlim([-reach, reach])
        ax.set_ylim([-reach, reach])
        ax.set_zlim([-0.5, reach])        # extend axis down so pedestal is visible
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])

        # --- Physical pedestal (cylinder on ground, top at z=0 = joint 1 axis) ---
        base_radius = 0.15
        pedestal_height = 0.24             # how far below joint 1 the floor is
        phi = np.linspace(0, 2*np.pi, 40)
        z = np.linspace(-pedestal_height, 0, 2)
        PHI, Z = np.meshgrid(phi, z)
        X = base_radius * np.cos(PHI)
        Y = base_radius * np.sin(PHI)
        ax.plot_surface(X, Y, Z, color='gray', alpha=0.5, linewidth=0)

        # Top cap (where the arm emerges)
        cx = base_radius * np.cos(phi)
        cy = base_radius * np.sin(phi)
        ax.plot(cx, cy, np.zeros_like(phi), color='dimgray', linewidth=2)

        # Floor circle (for visual reference)
        ax.plot(cx, cy, -pedestal_height * np.ones_like(phi),
                color='dimgray', linewidth=2)

        # Floor plane (optional light grid reference)
        floor_size = 0.8
        xx, yy = np.meshgrid([-floor_size, floor_size], [-floor_size, floor_size])
        ax.plot_surface(xx, yy, -pedestal_height * np.ones_like(xx),
                        color='lightgray', alpha=0.15, linewidth=0)

    def _draw(self, ax, theta, title=''):
        pts = self.joint_positions(theta)
        # Links
        line, = ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        '-o', color='steelblue', linewidth=3,
                        markersize=7, markerfacecolor='orange',
                        markeredgecolor='black')
        # End-effector as a distinct marker
        ee, = ax.plot([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]],
                      marker='*', markersize=15, color='red',
                      markeredgecolor='black')
        # Base
        ax.scatter([0], [0], [0], c='black', s=60, marker='s')
        if title:
            ax.set_title(title)
        return line, ee

    def show_pose(self, theta, title='Pose'):
        """Display a single static pose."""
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)
        self._draw(ax, np.asarray(theta), title=title)
        plt.tight_layout()
        plt.show()

    def animate(self, theta_traj, dt=0.02, title='Trajectory', save_as=None):
        """
        theta_traj : (N, n) array of joint configurations over time
        dt         : seconds per frame (controls playback speed)
        save_as    : optional filename (e.g. 'arm.gif' or 'arm.mp4')
        """
        theta_traj = np.asarray(theta_traj)
        N = theta_traj.shape[0]

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)

        # Initialize artists
        pts0 = self.joint_positions(theta_traj[0])
        line, = ax.plot(pts0[:, 0], pts0[:, 1], pts0[:, 2],
                        '-o', color='steelblue', linewidth=3,
                        markersize=7, markerfacecolor='orange',
                        markeredgecolor='black')
        ee_marker, = ax.plot([pts0[-1, 0]], [pts0[-1, 1]], [pts0[-1, 2]],
                             marker='*', markersize=15, color='red',
                             markeredgecolor='black')
        ee_trail, = ax.plot([], [], [], '-', color='red', alpha=0.4, linewidth=1)
        ax.scatter([0], [0], [0], c='black', s=60, marker='s')
        ax.set_title(title)

        trail = [[], [], []]

        def update(frame):
            pts = self.joint_positions(theta_traj[frame])
            line.set_data(pts[:, 0], pts[:, 1])
            line.set_3d_properties(pts[:, 2])
            ee_marker.set_data([pts[-1, 0]], [pts[-1, 1]])
            ee_marker.set_3d_properties([pts[-1, 2]])
            trail[0].append(pts[-1, 0])
            trail[1].append(pts[-1, 1])
            trail[2].append(pts[-1, 2])
            ee_trail.set_data(trail[0], trail[1])
            ee_trail.set_3d_properties(trail[2])
            ax.set_title(f'{title}  frame {frame+1}/{N}')
            return line, ee_marker, ee_trail

        anim = FuncAnimation(fig, update, frames=N,
                             interval=dt*1000, blit=False, repeat=True)

        if save_as:
            if save_as.endswith('.gif'):
                anim.save(save_as, writer='pillow', fps=int(1/dt))
            else:
                anim.save(save_as, fps=int(1/dt))

        plt.tight_layout()
        plt.show()
        return anim     # keep a reference or the animation gets garbage-collected