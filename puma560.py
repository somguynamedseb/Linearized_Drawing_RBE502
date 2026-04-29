import numpy as np

class Puma560:
    def __init__(self):
        self.jNum = 6  # number of joints

        # ============================================================
        # Kinematics extracted directly from the Webots PROTO file.
        # Frame: Webots WORLD frame, z-up, floor at z = 0.
        # Joint 1 (shoulder) sits at z = 0.669.
        # All values come from walking the PROTO's HingeJoint anchors
        # and endPoint translations down the tree.
        # ============================================================

        # Joint axis directions, verbatim from the PROTO.
        # At home (all parent rotations = 0) each link frame is aligned
        # with the world frame, so these are also the axes in space.
        self.w = np.array([
            [ 0, 0, 1],   # joint 1  shoulder      PROTO axis  0  0  1
            [-1, 0, 0],   # joint 2  upper arm     PROTO axis -1  0  0
            [-1, 0, 0],   # joint 3  elbow         PROTO axis -1  0  0
            [ 0, 0, 1],   # joint 4  wrist roll    PROTO axis  0  0  1
            [-1, 0, 0],   # joint 5  wrist pitch   PROTO axis -1  0  0
            [ 0, 0, 1],   # joint 6  tool roll     PROTO axis  0  0  1
        ], dtype=float)

        # A point on each joint axis at home, in the world frame.
        # Cumulative anchor walk:
        #   J1: shoulder anchor                       (0, 0, 0.669)
        #   J2: J1 + (-0.1622,  0,     0    )
        #   J3: J2 + ( 0.086,   0.42,  0    )
        #   J4: J3 + (-0.074,  -0.02,  0.353)
        #   J5: J4 + ( 0,       0,     0.079)
        #   J6: J5 + ( 0,       0,    -0.08 )
        q1 = np.array([ 0.0000, 0.000, 0.669])
        q2 = np.array([-0.1622, 0.000, 0.669])
        q3 = np.array([-0.0762, 0.420, 0.669])
        q4 = np.array([-0.1502, 0.400, 1.022])
        q5 = np.array([-0.1502, 0.400, 1.101])
        q6 = np.array([-0.1502, 0.400, 1.021])
        self.Q = np.array([q1, q2, q3, q4, q5, q6])
        
        joint_limits = [
            (-2.79,  2.79),
            (-3.93,  0.79),
            (-0.79,  3.93),
            (-1.92,  2.97),
            (-1.75,  1.75),
            (-4.64,  4.64),
        ]
        self.q_min = np.array([lo for lo, _ in joint_limits])
        self.q_max = np.array([hi for _, hi in joint_limits])

        # Screw axes in the space (world) frame at home: S_i = [w_i; -w_i x q_i]
        self.slist = np.zeros((6, 6))
        for i in range(6):
            self.slist[0:3, i] = self.w[i]
            self.slist[3:6, i] = -np.cross(self.w[i], self.Q[i])

        # End-effector home pose: joint-6 endpoint (tool flange) frame.
        # At home, R = I and p = q6.
        # NOTE: the gripper sits ~0.23 m further along the flange's +z;
        # if you want the gripper tip pose instead of the flange pose,
        # post-multiply M by that fixed offset.
        self.M = np.eye(4)
        self.M[0:3, 3] = np.copy(q6)

        # Skew-symmetric matrices of each w_i (used by FK and Jacobian)
        self.w_skew = np.zeros((6, 3, 3))
        for i in range(6):
            self.w_skew[i] = np.array([
                [            0, -self.w[i][2],  self.w[i][1]],
                [ self.w[i][2],             0, -self.w[i][0]],
                [-self.w[i][1],  self.w[i][0],             0],
            ])

        # ============================================================
        # Dynamics setup
        # WARNING: the mass/CoM/inertia values below are the textbook
        # PUMA 560 figures (Corke / Armstrong-Khatib-Burdick). They were
        # defined for the *old* home pose with the arm pointing straight
        # up. The Webots home is different (arm extended along +y), so
        # the CoM offsets are no longer in the right body-frame
        # directions. FK / IK will be correct, but inverse_dynamics(),
        # mass_matrix(), gravity_forces() and coriolis_matrix() will be
        # off until you re-derive these in the new pose. Webots itself
        # auto-computes mass/inertia from each link's boundingObject
        # when Physics{} is empty, so for controller work you may want
        # to either skip dynamics here or pull the inertia values from
        # Webots' supervisor API.
        # ============================================================
        mass = np.array([0.00, 17.40, 4.80, 0.82, 0.34, 0.09])
        com = np.array([[0.000, 0.068, 0.000, 0.000, 0.000, 0.000],
                        [0.000, 0.006,-0.070, 0.000, 0.000, 0.000],
                        [0.000,-0.016, 0.014,-0.019, 0.000, 0.032]])
        Ixx = np.array([0.000, 0.130, 0.066, 0.0018, 0.00030, 0.00015])
        Iyy = np.array([0.350, 0.524, 0.0125, 0.0018, 0.00030, 0.00015])
        Izz = np.array([0.000, 0.539, 0.086, 0.0013, 0.00040, 0.00004])

        self.Mlist_abs = []
        for i in range(6):
            com_space = self.Q[i] + com[:, i]
            Mi = np.eye(4); Mi[0:3, 3] = com_space
            self.Mlist_abs.append(Mi)

        self.Glist = [np.diag([Ixx[i], Iyy[i], Izz[i], mass[i], mass[i], mass[i]])
                    for i in range(6)]

        self.Alist = [self._adjoint(np.linalg.inv(self.Mlist_abs[i])) @ self.slist[:, i]
                    for i in range(6)]

        self.Mrel = []
        prev = np.eye(4)
        for i in range(6):
            self.Mrel.append(np.linalg.inv(prev) @ self.Mlist_abs[i])
            prev = self.Mlist_abs[i]

    def _skew(self, w):
        return np.array([[0,    -w[2],  w[1]],
                        [w[2],  0,    -w[0]],
                        [-w[1], w[0],  0   ]])

    def _ad_little(self, V):
        # 6x6 ad operator for bracket of twists
        w, v = V[0:3], V[3:6]
        mat = np.zeros((6,6))
        mat[0:3, 0:3] = self._skew(w)
        mat[3:6, 0:3] = self._skew(v)
        mat[3:6, 3:6] = self._skew(w)
        return mat

    def _exp6_screw(self, S, theta):
        # General screw exponential (works for any 6-vec screw)
        w, v = S[0:3], S[3:6]
        T = np.eye(4)
        if np.linalg.norm(w) < 1e-10:
            T[0:3, 3] = v * theta
            return T
        wh = self._skew(w)
        T[0:3, 0:3] = np.eye(3) + np.sin(theta)*wh + (1 - np.cos(theta))*wh @ wh
        G = np.eye(3)*theta + (1 - np.cos(theta))*wh + (theta - np.sin(theta))*wh @ wh
        T[0:3, 3] = G @ v
        return T

    def _adjoint(self, T):
        R, p = T[0:3, 0:3], T[0:3, 3]
        p_skew = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]])
        Ad = np.zeros((6,6))
        Ad[0:3, 0:3] = R
        Ad[3:6, 3:6] = R
        Ad[3:6, 0:3] = np.dot(p_skew, R)
        return Ad

    def _log6(self, T):
        # Inverse of the screw exponential: returns 6-vector twist V from T
        R, p = T[0:3, 0:3], T[0:3, 3]
        cos_th = np.clip((np.trace(R) - 1)/2, -1.0, 1.0)
        theta = np.arccos(cos_th)
        if theta < 1e-6:
            return np.concatenate([np.zeros(3), p])
        wh = (R - R.T) / (2*np.sin(theta))
        w = np.array([wh[2,1], wh[0,2], wh[1,0]])
        Ginv = np.eye(3)/theta - 0.5*wh + (1/theta - 0.5/np.tan(theta/2))*np.dot(wh, wh)
        v = np.dot(Ginv, p)
        return np.concatenate([w*theta, v*theta])

    def _exp6(self, i, theta_i):
        # Single-joint exponential — same math as one iteration of your FK loop
        T = np.eye(4)
        wh = self.w_skew[i]
        T[0:3, 0:3] = np.eye(3) + np.sin(theta_i)*wh + (1 - np.cos(theta_i))*np.dot(wh, wh)
        G = np.eye(3)*theta_i + (1 - np.cos(theta_i))*wh + (theta_i - np.sin(theta_i))*np.dot(wh, wh)
        T[0:3, 3] = np.dot(G, self.slist[3:6, i])
        return T

    def _space_jacobian(self, theta):
        Js = self.slist.copy().astype(float)
        T = np.eye(4)
        for i in range(1, self.jNum):
            T = np.dot(T, self._exp6(i-1, theta[i-1]))
            Js[:, i] = np.dot(self._adjoint(T), self.slist[:, i])
        return Js

    def FK(self,theta):
        e_w_theta = np.zeros((self.jNum,4,4))
        for i in range(self.jNum):
            e_w_theta[i] = np.eye(4)
            e_w_theta[i][0:3, 0:3] = np.eye(3) + np.sin(theta[i]) * self.w_skew[i] + (1 - np.cos(theta[i])) * np.dot(self.w_skew[i], self.w_skew[i])
            G = np.eye(3) * theta[i] + (1 - np.cos(theta[i])) * self.w_skew[i] + (theta[i] - np.sin(theta[i])) * np.dot(self.w_skew[i], self.w_skew[i])
            e_w_theta[i][0:3,3] = np.dot(G,self.slist[3:6,i])
            
        T = np.eye(4)
        for i in range(self.jNum):
            T = np.dot(T, e_w_theta[i])
        T = np.dot(T, self.M)
        return T

    def IK(self, T_sd, theta0=None, tol=1e-3, max_iter=100, lam=0.05,
        n_restarts=8, seed=0):
        """
        Damped least-squares IK with multi-start branch search.

        Tries seeds in this order:
        1. theta0 if provided (warm start from previous trajectory point)
        2. A "smart home" in the middle of joint range (elbow up)
        3. Hard zero (the kinematic home)
        4. Branch flips: shoulder-flipped, elbow-down, both flipped
        5. Random fillers within limits, until n_restarts seeds total

        Returns the first seed that converges. If none do, returns the
        best in-limits attempt by twist-norm residual.

        For trajectories, pass the previous frame's solution as theta0 --
        one seed is usually enough when consecutive targets are nearby.
        """
        n   = self.jNum
        rng = np.random.default_rng(seed)

        # Mid-range "elbow up" seed -- the single biggest reliability win,
        # because q=zeros sits on joint 2's max and joint 3's min.
        smart_home = np.array([0.0, -1.5,  1.5, 0.0, 0.0, 0.0])

        seeds = []
        if theta0 is not None:
            seeds.append(np.clip(np.asarray(theta0, dtype=float),
                                self.q_min, self.q_max))
        seeds.append(smart_home)
        seeds.append(np.zeros(n))                               # original home
        seeds.append(np.array([np.pi, -1.5,  1.5, 0.0, 0.0, 0.0]))  # shoulder flip
        seeds.append(np.array([0.0,   -1.5, -0.5, 0.0, 0.0, 0.0]))  # elbow down
        seeds.append(np.array([np.pi, -1.5, -0.5, 0.0, 0.0, 0.0]))  # both
        while len(seeds) < n_restarts:
            seeds.append(rng.uniform(self.q_min, self.q_max))

        best_q, best_err = None, np.inf
        for q_seed in seeds:
            q, ok = self._ik_single(T_sd, q_seed, tol, max_iter, lam)
            if ok:
                return q, True
            Vs  = self._log6(T_sd @ np.linalg.inv(self.FK(q)))
            err = np.linalg.norm(Vs)
            if err < best_err:
                best_err, best_q = err, q

        return best_q, False

    def _ik_single(self, T_sd, theta0, tol, max_iter, lam):
        """Single-start damped LS IK with joint-limit handling."""
        n   = self.jNum
        eps = 1e-9
        I6  = np.eye(6)
        theta = np.clip(np.asarray(theta0, dtype=float), self.q_min, self.q_max)

        for _ in range(max_iter):
            T_sb = self.FK(theta)
            Vs   = self._log6(T_sd @ np.linalg.inv(T_sb))
            if np.linalg.norm(Vs[:3]) < tol and np.linalg.norm(Vs[3:]) < tol:
                return theta, True

            Js     = self._space_jacobian(theta)
            dtheta = Js.T @ np.linalg.solve(Js @ Js.T + lam**2 * I6, Vs)

            at_low  = (theta <= self.q_min + 1e-9) & (dtheta < 0)
            at_high = (theta >= self.q_max - 1e-9) & (dtheta > 0)
            locked  = at_low | at_high
            if np.any(locked):
                free = ~locked
                if not np.any(free):
                    break
                Jf = Js[:, free]
                df = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam**2 * I6, Vs)
                dtheta = np.zeros(n)
                dtheta[free] = df

            alpha = 1.0
            for i in range(n):
                if   dtheta[i] >  eps:
                    alpha = min(alpha, (self.q_max[i] - theta[i]) / dtheta[i])
                elif dtheta[i] < -eps:
                    alpha = min(alpha, (self.q_min[i] - theta[i]) / dtheta[i])
            alpha = max(alpha, 0.0)

            theta = np.clip(theta + alpha * dtheta, self.q_min, self.q_max)

        return theta, False

    def inverse_dynamics(self, q, qd, qdd, g_vec=np.array([0, 0, -9.81]), Ftip=np.zeros(6)):
        """
        Newton-Euler inverse dynamics.
        Returns the joint torques tau that produce (q, qd, qdd).
        """
        n = self.jNum

        # Relative transforms at current q: T_{i+1, i}
        T_rel = []
        for i in range(n):
            T_rel.append(self._exp6_screw(-self.Alist[i], q[i]) @ np.linalg.inv(self.Mrel[i]))
        T_rel.append(np.eye(4))   # tip frame = last link frame (no EE offset)

        # Forward pass: twists and accelerations
        V  = [np.zeros(6) for _ in range(n+1)]
        Vd = [np.zeros(6) for _ in range(n+1)]
        Vd[0][3:6] = -g_vec        # gravity injected as base acceleration
        for i in range(n):
            AdT = self._adjoint(T_rel[i])
            V[i+1]  = AdT @ V[i]  + self.Alist[i] * qd[i]
            Vd[i+1] = AdT @ Vd[i] + self._ad_little(V[i+1]) @ self.Alist[i] * qd[i] \
                                + self.Alist[i] * qdd[i]

        # Backward pass: wrenches and torques
        F = [np.zeros(6) for _ in range(n+1)]
        F[n] = Ftip
        tau = np.zeros(n)
        for i in range(n-1, -1, -1):
            Gi = self.Glist[i]
            F[i] = self._adjoint(T_rel[i+1]).T @ F[i+1] + Gi @ Vd[i+1] \
                - self._ad_little(V[i+1]).T @ (Gi @ V[i+1])
            tau[i] = F[i] @ self.Alist[i]
        return tau

    def mass_matrix(self, q):
        n = self.jNum
        M = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n); ei[i] = 1
            M[:, i] = self.inverse_dynamics(q, np.zeros(n), ei, g_vec=np.zeros(3))
        return M

    def gravity_forces(self, q):
        return self.inverse_dynamics(q, np.zeros(self.jNum), np.zeros(self.jNum))

    def coriolis_matrix(self, q, qd, eps=1e-6):
        # Christoffel symbols with numerical dM/dq
        n = self.jNum
        M0 = self.mass_matrix(q)
        dM = np.zeros((n, n, n))
        for k in range(n):
            qk = q.copy(); qk[k] += eps
            dM[:, :, k] = (self.mass_matrix(qk) - M0) / eps

        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Gamma = 0.5 * (dM[i,j,k] + dM[i,k,j] - dM[j,k,i])
                    C[i, j] += Gamma * qd[k]
        return C

    def tau_example(self, q, qd, qdd):
        M = self.mass_matrix(q)           # 6x6
        C = self.coriolis_matrix(q, qd)   # 6x6
        g = self.gravity_forces(q)        # 6-vec
        return M @ qdd + C @ qd + g         # the dynamics equation