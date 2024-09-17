"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

from abc import abstractmethod
import numpy as np
import matplotlib.pylab as plt

E = np.array([[0, 1],[-1, 0]])

# ----------------------------------------------------------------------------
# GVF_traj common class
# ----------------------------------------------------------------------------

# The computation are vectorized to be performed with N agents.
# -------------------------------------------------------------
# -> p      (N x 2)
# -> XYoff  (1 x 2)
# -> R      (2 x 2) rotation matrix (its inverse is = to its transpose)

class gvf_traj:
    XYoff: list = [0,0]
    traj_points: list = None
    mapgrad_pos: list = None
    mapgrad_vec: list = None

    @abstractmethod
    def phi(self, p: np.ndarray) -> float:
        """
        Phi evaluation.
        """
        pass

    @abstractmethod
    def grad_phi(self, p: np.ndarray) -> np.ndarray:
        """
        Phi gradient.
        """
        pass

    @abstractmethod
    def hess_phi(self, p: np.ndarray) -> np.ndarray:
        """
        Phi hessian.
        """
        pass

    @abstractmethod
    def gen_param_points(self, pts: int) -> np.ndarray:
        pass

    def gen_vector_field(self, XYoff, area, s, ke, pts = 30):
        """
        Generate the vector field to be plotted.
        """
        x_lin = np.linspace(XYoff[0] - 0.5*np.sqrt(area), \
                            XYoff[0] + 0.5*np.sqrt(area), pts)
        y_lin = np.linspace(XYoff[1] - 0.5*np.sqrt(area), \
                            XYoff[1] + 0.5*np.sqrt(area), pts)
        mapgrad_X, mapgrad_Y = np.meshgrid(x_lin, y_lin)
        mapgrad_X = np.reshape(mapgrad_X, -1)
        mapgrad_Y = np.reshape(mapgrad_Y, -1)
        self.mapgrad_pos = np.array([mapgrad_X, mapgrad_Y]).T

        n = self.grad_phi(self.mapgrad_pos)
        t = s*n @ E.T

        e = self.phi(self.mapgrad_pos)[:,None]

        self.mapgrad_vec = t - ke*e*n

        norm = np.sqrt(self.mapgrad_vec[:,0]**2 + self.mapgrad_vec[:,1]**2)[:,None]
        self.mapgrad_vec = self.mapgrad_vec / norm

    def draw(self, fig=None, ax=None, xlim=None, ylim=None, draw_field=True, alpha=0.2, ls="--", lw=1, width=0.0025, color="k"):
        """
        Plot the trajectory and the vector field.
        """
        if fig == None:
            fig = plt.figure(dpi=100)
            ax = fig.subplots()
        elif ax == None:
            ax = fig.subplots()

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Get the trajectory points
        self.traj_points = self.gen_param_points()

        # Plot the trajectory
        ax.plot(self.XYoff[0], self.XYoff[1], "+k", zorder=0)
        traj, = ax.plot(self.traj_points[0], self.traj_points[1], c=color, ls=ls, lw=lw, zorder=0)

        # Plot the vector field
        if draw_field:
            if self.mapgrad_vec is not None:
                ax.quiver(self.mapgrad_pos[:,0], self.mapgrad_pos[:,1],
                          self.mapgrad_vec[:,0], self.mapgrad_vec[:,1],
                          alpha=alpha, width=width)
            else:
                print("Please run gen_vector_field() first.")
            
        return traj