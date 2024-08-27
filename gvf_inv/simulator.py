"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

from .gvf_traj import gvf_traj

# -90º rotation matrix
E = np.array([[0, 1],[-1, 0]])

# --------------------------------------------------------------------------------------

class simulator:
    def __init__(self, gvf_traj, x0, dt, s, ke, kn):

        # Simulator settings
        self.t = 0
        self.dt = dt

        self.p = x0[0]
        self.v = x0[1]
        self.phi = x0[2]

        self.N = self.p.shape[0]

        # GVF settings
        self.traj = gvf_traj

        self.s = s
        self.ke = ke
        self.kn = kn

        # Simulator data
        self.data = {
                "t": [],
                "p": [],
                "phi": [],
                "omega": [],
        }

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        self.data["t"].append(self.t)
        self.data["p"].append(self.p)
        self.data["phi"].append(self.phi)
        self.data["omega"].append(self.omega)

    def gvf_control(self):
        """
        Funtion to compute omega_gvf
        """
        # GVF trajectory data
        self.e = self.traj.phi(self.p) # Phi (error)  N x 1
        n = self.traj.grad_phi(self.p) # Phi gradient N x 2
        H = self.traj.hess_phi(self.p) # Phi hessian  N x 2 x 2
        t = self.s*n @ E.T              # Phi tangent  N x 2

        # Compute the desired angular velocity to aling with the vector field
        omega = np.zeros(self.N)
        for i in range(self.N):
            ei = self.e[i]
            ni = n[i,:][:,None]
            ti = t[i,:][:,None]

            if len(H.shape) < 3:
                Hi = H
            else:
                Hi = H[i,:,:]
            
            pd_dot = ti - self.ke*ei*ni

            norm_pd_dot = np.sqrt(pd_dot[0]**2 + pd_dot[1]**2)
            md = pd_dot / norm_pd_dot

            Apd_dot_dot = - self.ke * ni.T @ pd_dot * ni
            Bpd_dot_dot = (E.T - self.ke * ei) @ Hi @ pd_dot
            pd_dot_dot = Bpd_dot_dot + Apd_dot_dot

            md_dot_const = md.T @ E.T @ pd_dot_dot / norm_pd_dot
            md_dot = E.T @ md * md_dot_const

            omega_d = md_dot.T @ E.T @ md
            mr = np.array([np.cos(self.phi[i]), np.sin(self.phi[i])])

            omega[i] = omega_d + self.kn * mr @ E.T @ md

        # # Clip the response of the controller by the mechanical saturation value 
        # # (YES, we can do this here)
        # omega = self.omega_clip(omega)

        return -omega


    def int_euler(self):
        """
        Funtion to integrate the simulation step by step using Euler
        """

        # Compute the GVF omega control law
        self.omega = self.gvf_control()

        # Integrate
        p_dot = self.v * np.array([np.cos(self.phi), np.sin(self.phi)]).T

        self.t = self.t + self.dt
        self.p = self.p + p_dot * self.dt
        self.phi = (self.phi + self.omega * self.dt) % (2*np.pi)

        # Update output data
        self.update_data()