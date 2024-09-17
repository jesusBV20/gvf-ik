"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

from .gvf_traj import gvf_traj

# -90º rotation matrix
E = np.array([[0, 1],[-1, 0]])

# --------------------------------------------------------------------------------------

class simulator:
    def __init__(self, gvf_traj, x0, dt, s, ke, kn, A_fd, omega_fd):

        # Simulator settings
        self.t = 0
        self.dt = dt

        self.p = x0[0]
        self.v = x0[1]
        self.theta = x0[2]

        self.N = self.p.shape[0]

        # GVF settings
        self.traj = gvf_traj

        self.s = s
        self.ke = ke
        self.kn = kn

        self.A_fd = A_fd
        self.omega_fd = omega_fd

        self.phi = np.zeros(self.N)
        self.e = np.zeros(self.N)

        self.omega = np.zeros(self.N)

        self.p_pred = np.copy(self.p)
        self.pd_dot = None

        # Simulator data
        self.data = {
                "t": [],
                "p": [],
                "theta": [],
                "phi": [],
                "e": [],
                "p_pred": [],
                "omega_d": [],
                "omega_gvf": [],
                "omega": [],
                "n_norm": [],
                "t_norm": [],
        }

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        self.data["t"].append(self.t)
        self.data["p"].append(self.p.copy())
        self.data["theta"].append(self.theta.copy())
        self.data["phi"].append(self.phi.copy())
        self.data["e"].append(self.e.copy())
        self.data["p_pred"].append(self.p_pred.copy())

        self.data["omega_d"].append(self.omega_d.copy())
        self.data["omega_gvf"].append(self.omega_gvf.copy())
        self.data["omega"].append(self.omega.copy())

        self.data["n_norm"].append(self.n_norm.copy())
        self.data["t_norm"].append(self.t_norm.copy())

    def clip_omega(self, omega):
        """
        Function to bound the value of omega
        """
        max_w = np.pi/2
        w_clipped = np.copy(omega)
        for i in range(self.N):
            w_clipped[i] = np.clip(omega[i], -max_w, max_w)
        return w_clipped

    def gvf_check_alpha(self, J1, J2, phi, v):

        J_Jt = (J1*J1 + J2*J2)

        # 2. Compute th feedforward error
        e = phi + self.A_fd * np.sin(self.omega_fd * self.t)
        e_tdot = self.omega_fd * self.A_fd * np.cos(self.omega_fd * self.t)

        # 3. Compute the input term of p_dot (normal term)
        u = - self.ke * e

        un_x = J1 / J_Jt * (u - e_tdot)
        un_y = J2 / J_Jt * (u - e_tdot)

        un_norm2 = un_x*un_x + un_y*un_y

        return un_norm2 < v**2

    def gvf_control_IK_pred(self):
        # 1. Collect GVF trajectory data
        phi = self.traj.phi(self.p_pred)    # Phi value    N x 1
        J = self.traj.grad_phi(self.p_pred) # Phi gradient N x 2

        for i in range(self.N):

            J1 = J[i,0]
            J2 = J[i,1]
            
            J_Jt = (J1*J1 + J2*J2)

            cond_flag = self.gvf_check_alpha(J1, J2, phi[i], self.v[i])

            # 2. Compute th feedforward error
            if cond_flag:
                e = phi[i] + self.A_fd * np.sin(self.omega_fd * self.t)
                e_tdot = self.omega_fd * self.A_fd * np.cos(self.omega_fd * self.t)
            else:
                e = phi[i]
                e_tdot = 0

            # 3. Compute the input term of p_dot (normal term)
            u = - self.ke * e

            un_x = J1 / J_Jt * (u - e_tdot)
            un_y = J2 / J_Jt * (u - e_tdot)

            un_norm2 = un_x*un_x + un_y*un_y
            un_norm = np.sqrt(un_norm2)

            # 4. Compute alpha and the tangent term of p_dot
            ut_x = self.s * J2
            ut_y = -self.s * J1

            ut_norm = np.sqrt(ut_x**2 + ut_y**2)
            
            ut_hat_x = ut_x / ut_norm
            ut_hat_y = ut_y / ut_norm
            
            # 5. Compute alpha and p_dot
            if cond_flag:
                alpha = np.sqrt(self.v[i]**2 - un_norm2)

                pd_dot_x = alpha * ut_hat_x + un_x
                pd_dot_y = alpha * ut_hat_y + un_y
            else:
                alpha = 0
                
                pd_dot_x = self.v[i] * un_x / un_norm
                pd_dot_y = self.v[i] * un_y / un_norm

        # TODO: fix, just consider the last agent right now
        if cond_flag:
            self.pd_dot = np.array([pd_dot_x, pd_dot_y])
        else:
            self.pd_dot = None

    def gvf_control_IK(self):
        # 1. Collect GVF trajectory data
        phi = self.traj.phi(self.p)    # Phi value    N x 1
        J = self.traj.grad_phi(self.p) # Phi gradient N x 2
        H = self.traj.hess_phi(self.p) # Phi hessian  N x 2 x 2

        omega = np.zeros_like(self.omega)
        self.omega_d = np.zeros_like(self.omega)
        self.n_norm = np.zeros_like(self.omega)
        self.t_norm = np.zeros_like(self.omega)
        for i in range(self.N):

            J1 = J[i,0]
            J2 = J[i,1]
            
            if len(H.shape) < 3:
                H = np.array([H])
                
            H11 = H[i,0,0]
            H12 = H[i,0,1]
            H21 = H[i,1,0]
            H22 = H[i,1,1]

            J_Jt = (J1*J1 + J2*J2)

            cond_flag = self.gvf_check_alpha(J1, J2, phi[i], self.v[i])

            # 2. Compute th feedforward error
            if cond_flag:
                e = phi[i] + self.A_fd * np.sin(self.omega_fd * self.t)
                e_tdot = self.omega_fd * self.A_fd * np.cos(self.omega_fd * self.t)
                e_tddot = - self.omega_fd ** 2 * self.A_fd * np.sin(self.omega_fd * self.t)
            else:
                e = phi[i]
                e_tdot = 0
                e_tddot = 0

            # 3. Compute the input term of p_dot (normal term)
            u = - self.ke * e

            un_x = J1 / J_Jt * (u - e_tdot)
            un_y = J2 / J_Jt * (u - e_tdot)

            un_norm2 = un_x*un_x + un_y*un_y
            un_norm = np.sqrt(un_norm2)

            # 4. Compute alpha and the tangent term of p_dot
            ut_x = self.s * J2
            ut_y = -self.s * J1

            ut_norm = np.sqrt(ut_x**2 + ut_y**2)
            
            ut_hat_x = ut_x / ut_norm
            ut_hat_y = ut_y / ut_norm
            
            # 5. Compute alpha and p_dot
            if cond_flag:
                alpha = np.sqrt(self.v[i]**2 - un_norm2)

                pd_dot_x = alpha * ut_hat_x + un_x
                pd_dot_y = alpha * ut_hat_y + un_y
            else:
                alpha = 0
                
                pd_dot_x = self.v[i] * un_x / un_norm
                pd_dot_y = self.v[i] * un_y / un_norm

            ut_dot_x = self.s * (H12 * pd_dot_x + H22 * pd_dot_y)
            ut_dot_y = - self.s * (H11 * pd_dot_x + H21 * pd_dot_y) 

            # 8. Compute un_dot
            u_dot = - self.ke * (J1*pd_dot_x + J2*pd_dot_y + e_tdot)

            un_dot_A_x = (pd_dot_x*H11 + pd_dot_y*H21)
            un_dot_A_y = (pd_dot_x*H12 + pd_dot_y*H22)

            B_term = 2 * ((H11*pd_dot_x + H21*pd_dot_y)*J1 + (H21*pd_dot_x + H22*pd_dot_y)*J2) / J_Jt
            un_dot_B_x = - J1 * B_term
            un_dot_B_y = - J2 * B_term

            C_term = (u_dot - e_tddot) / J_Jt
            un_dot_C_x = J1 * C_term
            un_dot_C_y = J2 * C_term

            # print(un_dot_A_x, un_dot_B_x, un_dot_C_x, un_dot_D_x)

            un_dot_x = (un_dot_A_x + un_dot_B_x) * (u - e_tdot) / J_Jt + un_dot_C_x
            un_dot_y = (un_dot_A_y + un_dot_B_y) * (u - e_tdot) / J_Jt + un_dot_C_y

            # 9. Compute omega_d and omega
            if cond_flag:
                alpha_dot = - (un_x*un_dot_x + un_y*un_dot_y) / (alpha)
                Bpd_ddot_x = alpha * (ut_dot_x / ut_norm + (ut_x*ut_x*ut_dot_x + ut_x*ut_y*ut_dot_y) / ut_norm**3)
                Bpd_ddot_y = alpha * (ut_dot_y / ut_norm + (ut_x*ut_y*ut_dot_x + ut_y*ut_y*ut_dot_y) / ut_norm**3)

                pd_ddot_x = alpha_dot * ut_hat_x + Bpd_ddot_x + un_dot_x
                pd_ddot_y = alpha_dot * ut_hat_y + Bpd_ddot_y + un_dot_y
            else:
                pd_ddot_x = self.v[i] * (un_dot_x / un_norm + (un_x*un_x*un_dot_x + un_x*un_y*un_dot_y) / un_norm / un_norm2)
                pd_ddot_y = self.v[i] * (un_dot_y / un_norm + (un_x*un_y*un_dot_x + un_y*un_y*un_dot_y) / un_norm / un_norm2)
            
            omega_d = - (- pd_dot_x*pd_ddot_y + pd_dot_y*pd_ddot_x) / (self.v[i]*self.v[i])
            self.omega_d[i] = omega_d

            r_x = self.v[i] * np.cos(self.theta[i])
            r_y = self.v[i] * np.sin(self.theta[i])

            # print(omega_d, self.kn * (pd_dot_x*r_y - pd_dot_y*r_x))

            omega[i] = omega_d - self.kn * (pd_dot_x*r_y - pd_dot_y*r_x) / (self.v[i]*self.v[i])

            # Save data ----------------------
            self.phi[i] = phi[i]
            self.e[i] = e
            self.n_norm[i] = un_norm
            self.t_norm[i] = alpha
            # ---------------------------------

        return omega

    def int_euler(self):
        """
        Funtion to integrate the simulation step by step using Euler
        """

        # Compute the GVF omega control law
        self.omega_gvf = self.gvf_control_IK()
        self.omega = self.clip_omega(self.omega_gvf)

        # Integrate
        p_dot = self.v * np.array([np.cos(self.theta), np.sin(self.theta)]).T

        self.t = self.t + self.dt
        self.p = self.p + p_dot * self.dt
        self.theta = (self.theta + self.omega * self.dt) % (2*np.pi)

        if self.pd_dot is not None:
            self.gvf_control_IK_pred()
            self.p_pred = self.p_pred + self.pd_dot * self.dt
        else:
            self.p_pred = np.copy(self.p)

        # Update output data
        self.update_data()