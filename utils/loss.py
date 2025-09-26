# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Binary Cross Entropy (BCE) loss
"""

import numpy as np


class BCELoss:
    """
    Binary cross-entropy (BCE) loss function with regularization. The loss is
    computed between the observations and the likelihood function of the interpolated
    model. The regularization term is a penalty on the squared difference between
    consecutive interpolated values.

    Parameters
    ----------

        t : `np.ndarray`
            Time samples

        x : `np.ndarray`
            Observations

        likelihood_fun : `utils.Sigmoid`
            Likelihood function

        interp_fun : `utils.Interpolator`
            Interpolation function

        beta_regularization : `float` (default: 0.)
            Regularization parameter for temporal smoothness

        weights : `np.ndarray` (default: None)
            Weights multiplying the BCE at different time samples. If `None`,
            all weights are set to 1.

    Inputs
    ------

        knots : `np.ndarray`
            Knots

        f_knots : `np.ndarray`
            Interpolated knots

    Outputs
    -------

        loss : `float`
            Cross-entropy loss

        gradient : `np.ndarray`
            Gradient of the loss wrt f_knots

    """

    def __init__(self,
                 t,
                 x,
                 likelihood_fun,
                 interp_fun,
                 beta_regularization=0.,
                 weights=None):
        # Time samples
        self.t = np.array(t)
        # Observations
        self.x = np.array(x)
        # Likelihood function
        self.likelihood_fun = likelihood_fun
        # Interpolation function
        self.interp_fun = interp_fun
        # Regularization parameter:
        # Multiplies the squared difference (f_knots[k] - f_knots[k-1])**2
        self.beta_regularization = beta_regularization
        # weights
        if weights is None:
            self.weights = np.ones_like(self.x)
        else:
            self.weights = weights

    def __call__(self,
                 knots,
                 f_knots):
        knots = np.array(knots)
        f_knots = np.array(f_knots)
        # Piece-wise linear interpolation from the model
        f_interp = self.interp_fun(knots, f_knots, self.t)

        # Likelihood given the model
        p_pred = self.likelihood_fun(f_interp)

        # Cross-entropy loss
        ce_loss = -np.sum(self.weights *
                          (self.x * np.log(p_pred) + (1 - self.x) * np.log(1 - p_pred)))

        # Regularization term: Sums the squared difference between consecutive f_knots
        reg_loss = self.beta_regularization * \
            np.sum((f_knots[1:] - f_knots[:-1])**2)
        return ce_loss + reg_loss

    def gradient(self,
                 knots,
                 f_knots):
        """
        Compute gradient of CE loss wrt f_knot

        Input
        -----

        knots : `np.ndarray`
            Knot positions

        f_knots : `np.ndarray`
            Function values at knots

        Output
        ------

        d_loss_d_fknot : `np.ndarray`
            Gradient of loss wrt f_knots
        """
        # Piece-wise linear interpolation from the model
        f_interp = self.interp_fun(knots, f_knots, self.t)

        # Likelihood given the model
        p_pred = self.likelihood_fun(f_interp)

        # Gradient of the interpolation function wrt f_knots
        d_f_interp_d_f_knot = self.interp_fun.d_f_interp_d_f_knot(
            knots, f_knots, self.t)

        # Gradient of the likelihood wrt f_interp
        d_likelihood_d_finterp = self.likelihood_fun.derivative(f_interp)

        # Gradient of loss wrt likelihood
        d_bce_d_likelihood = (1 - self.x[:, np.newaxis]) / \
            (1 - p_pred[:, np.newaxis]) \
            - self.x[:, np.newaxis] / p_pred[:, np.newaxis]

        # Gradient of loss wrt f_knot via chain rule
        d_bce_d_fknot = (self.weights[:, np.newaxis] * d_bce_d_likelihood *
                         d_likelihood_d_finterp[:, np.newaxis] *
                         d_f_interp_d_f_knot).sum(axis=0)

        # Gradient of regularization term wrt f_knot
        d_reg_d_fknot = np.r_[(f_knots[0] - f_knots[1]),
                              (2*f_knots[1:-1] - f_knots[:-2] - f_knots[2:]),
                              (f_knots[-1] - f_knots[-2])]
        d_reg_d_fknot = 2 * self.beta_regularization * d_reg_d_fknot

        # Gradient of loss wrt f_knots
        d_loss_d_fknot = d_bce_d_fknot + d_reg_d_fknot

        return d_loss_d_fknot
