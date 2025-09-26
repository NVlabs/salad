# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
BLER and SINR interpolators, used for SALAD's SINR teacher inference
"""

import numpy as np


class Sigmoid:
    """
    Implements a sigmoid likelihood function with configurable center and scale
    parameters.

    Parameters
    ----------
        center: `float` (default: 0.)
            Center of the sigmoid function

        scale: `float` (default: 1.)
            Scale of the sigmoid function, determining its steepness

        clip_val: `float` (default: 1e-16)
            Clip value for the sigmoid function. Useful to avoid numerical
            errors

    Output
    ------
        val: `float`
            Sigmoid function value
    """

    def __init__(self, center=0., scale=1., clip_val=1e-16):
        self.center = center
        self.scale = scale
        self.clip_val = clip_val

    def __call__(self, x):
        val = 1 / (1 + np.exp(-(x - self.center) / self.scale))
        return np.clip(val, self.clip_val, 1 - self.clip_val)

    def derivative(self, x):
        """Compute the derivative of the sigmoid function with respect to x."""
        return self(x) * (1 - self(x)) / self.scale


class Interpolator:
    """
    Base class for interpolators

    Input:
    ------

        knots: (n_knots,)
            Knots of the spline

        f_knots: (n_knots,)
            Values of the function at the knots

        t: (n_samples,)
            Time points at which to evaluate the interpolation

    Output:
    -------

        f_interp: (n_samples,)
            Interpolated function values at time t

    """
    @staticmethod
    def get_knots(n_knots, support):
        """ Get the knots for the interpolation """

    def d_f_interp_d_f_knot(self, knots, f_knots, t):
        """
        Evaluate the gradient of the interpolation function at time t with
        respect to each knot value 
        """

    def __call__(self, knots, f_knots, t):
        """ Interpolate the function at time t """


class Spline1stOrder(Interpolator):
    """
    Piece-wise linear, i.e., 1st order spline, interpolation

    Input:
    ------

        knots: (n_knots,)
            Knots of the spline

        f_knots: (n_knots,)
            Values of the function at the knots

        t: (n_samples,)
            Time points at which to evaluate the interpolation

    Output:
    -------

        f_interp: (n_samples,)
            Interpolated function values at time t
    """

    @staticmethod
    def get_knots(n_knots, support):
        """ Get the knots for the interpolation """
        return np.linspace(*support, n_knots)

    def d_f_interp_d_f_knot(self, knots, f_knots, t):
        """
        Evaluate the gradient of the spline interpolation at time t with
        respect to each knot value 

        Output:
        -------
            d_interp_d_fknot: (len(t), n_knots)
                Array of the gradient of the spline interpolation at time t with
                respect to each knot value 
        """
        # Get the index of the points between the knots
        t_idx = (t[np.newaxis, :] >= knots[:-1, np.newaxis]).sum(axis=0) - 1

        knots_idx = np.arange(len(knots))[np.newaxis, :]

        tmp = (t - knots[t_idx]) / (knots[t_idx+1] - knots[t_idx])
        tmp = tmp[:, np.newaxis]
        # Derivative of the interpolation function wrt to f(knots)
        # [n_samples, n_knots]
        t_idx = t_idx[:, np.newaxis]
        der = (t_idx == knots_idx) * (1 - tmp) + \
            (t_idx == (knots_idx - 1)) * tmp
        return der

    def __call__(self, knots, f_knots, t):
        t = [t] if not hasattr(t, '__len__') else t
        # Get the index of the points between the knots
        t_idx = (t[:, np.newaxis] >=
                 knots[np.newaxis, :-1]).sum(axis=1) - 1

        # Piece-wise linear interpolation
        delta_f_knots = f_knots[t_idx+1] - f_knots[t_idx]
        delta_knots = knots[t_idx+1] - knots[t_idx]
        f_interp = delta_f_knots / delta_knots * \
            (t - knots[t_idx]) + f_knots[t_idx]
        return f_interp


class LagrangianInterpolator(Interpolator):
    """
    Lagrangian interpolation

    Input:
    ------

        knots: (n_knots,)
            Knots of the spline

        f_knots: (n_knots,)
            Values of the function at the knots

        t: (n_samples,)
            Time points at which to evaluate the interpolation

    Output:
    -------

        f_interp: (n_samples,)
            Interpolated function values at time t
    """
    @staticmethod
    def get_knots(n_knots, support):
        """ Get the knots for the interpolation """
        # Chebyshev knots to avoid wild oscillations at the interval boundaries
        knots = np.cos(np.arange(n_knots) * np.pi / (n_knots-1))
        # Rescale to the interval [min(t), max(t)]
        knots = (knots + 1) / 2 * (support[1] - support[0]) + support[0]
        return knots

    def d_f_interp_d_f_knot(self, knots, f_knots, t):
        """
        Evaluate the gradient of the Lagrangian interpolation at time t with
        respect to each knot value 

        Output:
            d_interp_d_fknot: (len(t), n_knots) array of the gradient of the
                Lagrangian interpolation at time t with respect to each knot value
        """
        n_knots = f_knots.shape[-1]

        idx = np.arange(n_knots-1)[np.newaxis, :] + \
            np.triu(np.ones((n_knots-1, n_knots-1)))
        idx = np.concatenate([idx, [np.arange(n_knots-1)]], axis=0)
        idx = idx.astype(int)

        num = np.prod(np.array(t[:, np.newaxis, np.newaxis]
                               ) - knots[idx][np.newaxis, ::], axis=-1)
        den = np.prod(knots[np.newaxis, :, np.newaxis] -
                      knots[idx][np.newaxis, ::], axis=-1)
        # (len(t), n_knots)
        return num / den

    def __call__(self, knots, f_knots, t):
        t = [t] if not hasattr(t, '__len__') else t
        der = self.d_f_interp_d_f_knot(knots,
                                       f_knots,
                                       np.array(t))
        return np.sum(der * f_knots[np.newaxis, ::], axis=-1)
