import numpy as np

class Puma560:
    def __init__(self):
        # link lengths in meters
        a2 = 0.4318 
        a3 = 0.0203 
        d3 = 0.15005
        d4 = 0.4318 
        self.jNum = 6 # number of joints

        # unit vectors along the joint axes in the home configuration
        self.w = np.array([
                    [0,0,1],
                    [0,1,0],
                    [0,1,0],
                    [0,0,1],
                    [0,1,0],
                    [0,0,1]])

        # position of each joint in the home configuration
        q1 = np.array([0, 0,0])
        q2 = np.array([0, 0, 0])
        q3 = np.array([0, d3, a2])
        q4 = np.array([a3, d3, a2+d4])
        q5 = q4
        q6 = q4
        Q  = np.array([q1, q2, q3, q4, q5, q6])

        # screw axes in space frame when the manipulator is at home position
        self.slist = np.zeros((6,6))
        for i in range(6):
            self.slist[0:3,i] = self.w[i]
            self.slist[3:6,i] = -np.cross(self.w[i], Q[i])

        #home matrix configuration of the end-effector
        self.M = np.eye(4)
        self.M[0:3, 3] = [a3, d3, a2+d4] 
        
        #skew symmetric matrix of w
        self.w_skew = np.zeros((6,3,3))
        for i in range(6):
            self.w_skew[i] = np.array([ [0, -self.w[i][2], self.w[i][1]],
                                        [self.w[i][2], 0, -self.w[i][0]],
                                        [-self.w[i][1], self.w[i][0], 0]])
        
        
        
        # --- Dynamics setup ---
        # Link i's body frame: at its CoM, axes aligned with space frame at home
        mass = np.array([0.00, 17.40, 4.80, 0.82, 0.34, 0.09])
        com = np.array([[0.000, 0.068, 0.000, 0.000, 0.000, 0.000],
                        [0.000, 0.006,-0.070, 0.000, 0.000, 0.000],
                        [0.000,-0.016, 0.014,-0.019, 0.000, 0.032]])
        Ixx = np.array([0.000, 0.130, 0.066, 0.0018, 0.00030, 0.00015])
        Iyy = np.array([0.350, 0.524, 0.0125, 0.0018, 0.00030, 0.00015])
        Izz = np.array([0.000, 0.539, 0.086, 0.0013, 0.00040, 0.00004])

        # Home configs of each link frame in space (frame at CoM)
        self.Mlist_abs = []
        for i in range(6):
            com_space = Q[i] + com[:, i]
            Mi = np.eye(4); Mi[0:3, 3] = com_space
            self.Mlist_abs.append(Mi)

        # Spatial inertias G_i (6x6) in each link frame
        self.Glist = [np.diag([Ixx[i], Iyy[i], Izz[i], mass[i], mass[i], mass[i]])
                    for i in range(6)]

        # Body-frame screw axes: A_i = Ad_{M_i^-1} S_i
        self.Alist = [self._adjoint(np.linalg.inv(self.Mlist_abs[i])) @ self.slist[:, i]
                    for i in range(6)]

        # Relative transforms M_{i-1, i} (link i-1 frame → link i frame at home)
        self.Mrel = []
        prev = np.eye(4)
        for i in range(6):
            self.Mrel.append(np.linalg.inv(prev) @ self.Mlist_abs[i])
            prev = self.Mlist_abs[i]

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
    
    def IK(self, T_sd, theta0 = np.array([0,0,0,0,0,0]), tol=1e-3, max_iter=20):
        theta = np.array(theta0, dtype=float)
        for _ in range(max_iter):
            T_sb = self.FK(theta)
            Vs = self._log6(np.dot(T_sd, np.linalg.inv(T_sb)))
            if np.linalg.norm(Vs[:3]) < tol and np.linalg.norm(Vs[3:]) < tol:
                return theta, True
            Js = self._space_jacobian(theta)
            theta = theta + np.dot(np.linalg.pinv(Js), Vs)
        return theta, False

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