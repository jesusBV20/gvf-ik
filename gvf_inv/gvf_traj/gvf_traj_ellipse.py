"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

from .gvf_traj import *

E = np.array([[0, 1],[-1, 0]])

# ----------------------------------------------------------------------------
# The computation are vectorized to be performed with N agents.
# -------------------------------------------------------------
# -> p      (N x 2)
# -> XYoff  (1 x 2)
# -> R is a rotation matrix (its inverse is equivalent to its transpose)
# ----------------------------------------------------------------------------

class gvf_ellipse(gvf_traj):
  def __init__(self, XYoff, alpha, a, b):
    super().__init__()

    # Ellipse parameters
    self.XYoff = XYoff
    self.alpha = alpha
    self.a, self.b = a, b

    self.cosa, self.sina = np.cos(alpha), np.sin(alpha)
    self.R = np.array([[self.cosa, self.sina], [-self.sina, self.cosa]])

    # Get the rajectory points
    self.traj_points = self.param_points()

  """\
  Function to cumpute the trajectory points
  """
  def param_points(self, pts = 100):
    t = np.linspace(0, 2*np.pi, pts)
    x = self.XYoff[0] + self.a * np.cos(-self.alpha) * np.cos(t) \
                      - self.b * np.sin(-self.alpha) * np.sin(t)
    y = self.XYoff[1] + self.a * np.sin(-self.alpha) * np.cos(t) \
                      + self.b * np.cos(-self.alpha) * np.sin(t)
    return [x, y]

  """\
  Phi(p)
  """
  def phi(self, p):
    w = self.XYoff * np.ones([p.shape[0],1])
    pel = (p - w) @ self.R
    return (pel[:,0]/self.a)**2 + (pel[:,1]/self.b)**2 - 1

  """\
  Phi gradiant
  """
  def grad_phi(self, p):
    w = self.XYoff * np.ones([p.shape[0],1])
    pel = (p - w) @ self.R
    return 2 * pel / [self.a**2, self.b**2] @ self.R.T


  """\
  Hessian
  """
  def hess_phi(self, p):
    H = np.zeros((2,2))
    H[0,0] = 2 * (self.cosa**2 / self.a**2 + self.sina**2 / self.b**2)
    H[0,1] = 2 * self.sina * self.cosa * (1 / self.b**2 - 1 / self.a**2)
    H[1,0] = H[0,0]
    H[1,1] = 2 * (self.sina**2 / self.a**2 + self.cosa**2 / self.b**2)
    return H

  """\
  Funtion to generate the vector field to be plotted
  """
  def vector_field(self, XYoff, area, s, ke, kr = 1, pts = 30):
    x_lin = np.linspace(XYoff[0] - 0.5*np.sqrt(area), \
                        XYoff[0] + 0.5*np.sqrt(area), pts)
    y_lin = np.linspace(XYoff[1] - 0.5*np.sqrt(area), \
                        XYoff[1] + 0.5*np.sqrt(area), pts)
    mapgrad_X, mapgrad_Y = np.meshgrid(x_lin, y_lin)
    mapgrad_X = np.reshape(mapgrad_X, -1)
    mapgrad_Y = np.reshape(mapgrad_Y, -1)
    self.mapgrad_pos = np.array([mapgrad_X, mapgrad_Y]).T

    w = self.XYoff * np.ones([self.mapgrad_pos.shape[0],1])
    pel = (self.mapgrad_pos - w) @ self.R

    n = self.grad_phi(self.mapgrad_pos)
    t = s*n @ E.T

    e = self.phi(self.mapgrad_pos)[:,None]

    self.mapgrad_vec = t - ke*e*n

    norm = np.sqrt(self.mapgrad_vec[:,0]**2 + self.mapgrad_vec[:,1]**2)[:,None]
    self.mapgrad_vec = self.mapgrad_vec / norm

