# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Self-Adaptive Link ADaptation (SALAD)
"""

import numpy as np
from algos import LinkAdaptation
from utils import StudentSINREstimator, TeacherSINREstimator


class SALAD(LinkAdaptation):
    """
    Self-Adaptive Link ADaptation (SALAD)

    Parameters
    ----------
        bler_target : `float`
            BLER target

        bler_sigmoid_params : `dict`
            Sigmoid parameters approximating the BLER function

        sinr_estimator : `StudentSINREstimator` | `TeacherSINREstimator`
            SINR estimator

        rho_calibration : `float` (default: 0.25)
            If the bias score is larger than this threshold, probing is allowed

        prob_probe : `float` (default: 0.15)
            Probability of probing, given that the bias score exceeds
            `prob_probe` 

        k_e : `float` (default: 0.02)
            Integral error gain

        score_window : `int` (default: 15)
            Window size for the bias score computation

        bler_target_probe : `float` (default: 0.95)
            Instantaneous BLER target for MCS probing

    Input
    -----
        is_nack : `list` of `int` | `int`
            Whether HARQ feedback is NACK

    Output
    ------
        mcs : `int`
            MCS index for the next transmission

    """

    def __init__(self,
                 bler_target: float,
                 bler_sigmoid_params: dict,
                 sinr_estimator: StudentSINREstimator | TeacherSINREstimator,
                 rho_calibration: float = 0.25,
                 prob_probe: float = .15,
                 k_e: float = .02,
                 score_window: int = 15,
                 bler_target_probe: float = .95):

        super().__init__(bler_target,
                         bler_sigmoid_params,
                         sinr_estimator=sinr_estimator)

        # Validate inputs
        if 'center' not in bler_sigmoid_params:
            raise ValueError('likelihood_params must contain a "center" key')
        if 'scale' not in bler_sigmoid_params:
            raise ValueError('likelihood_params must contain a "scale" key')
        if len(bler_sigmoid_params['center']) != len(bler_sigmoid_params['scale']):
            raise ValueError(
                'likelihood_params must contain the same number of "center" and "scale" values')

        self.prob_probe = prob_probe
        self.n_mcs = len(bler_sigmoid_params['center'])  # Number of MCS indices
        self.bler_sigmoid_params = bler_sigmoid_params
        self.bler_target = bler_target
        self.bler_target_probe = bler_target_probe
        self.rho_calibration = rho_calibration
        self.score_window = int(score_window)
        self.k_i = k_e

        # Initialize integral error
        self.integral_error = 0

        # Initialize history
        self.bler_est_hist = []
        self.is_probe_hist = []
        self.integral_error_hist = [self.integral_error]
        self.bias_score_ratio_hist = []  # ratio between bias score and its variance
        self.is_nack_hist = []
        self.tau_inst_hist = []

    def _process_harq_feedback(self,
                               is_nack):
        """
        Upon HARQ feedback, update integral error, SINR estimation, and ACK/NACK
        history 
        """
        # SINR estimation
        _, bler_est = self.sinr_estimator(
            is_nack,
            self.mcs_unackned[:len(is_nack)],
            return_bler=True)
        self.bler_est_hist.extend(bler_est)

        for is_nack_i in is_nack:

            # Integral error
            self.integral_error = self.integral_error + \
                (self.bler_target - is_nack_i)
            self.integral_error_hist.append(self.integral_error)

            # Update ACK/NACK history
            self.is_nack_hist.append(is_nack_i)
            if len(self.is_nack_hist) > self.score_window:
                self.is_nack_hist.pop(0)

    def _mcs_selection(self):
        """
        SALAD's MCS selection. The instantaneous BLER target is either the long-term
        BLER target or the BLER target for MCS probing, based on the value of
        the bias score ratio. The long-term BLER target is enforced via a
        feedback loop.
        """
        # Bias score ratio: ratio between bias score and its variance
        is_nack_hist = np.array(self.is_nack_hist)
        bler_score_window = np.array(
            self.bler_est_hist[-self.score_window:])
        if len(self.is_nack_hist) == 0:
            bias_score_ratio = 0
        else:
            bias_score = np.mean(bler_score_window - is_nack_hist)
            var_bias_score = np.sum(bler_score_window *
                                    (1 - bler_score_window)) / len(is_nack_hist)**2
            bias_score_ratio = bias_score / np.sqrt(var_bias_score)
        self.bias_score_ratio_hist.append(bias_score_ratio)

        # Decide whether to probe
        can_probe = (bias_score_ratio > self.rho_calibration) & \
            (np.random.rand() < self.prob_probe)

        # Instantaneous BLER target (tau_inst)
        if can_probe:
            # Probe a high instantaneous BLER target -> high MCS
            tau_inst = self.bler_target_probe
            is_probe = 1
        else:
            # No probing: use the long-term BLER target as instantaneous target
            tau_inst = self.bler_target
            is_probe = 0

        # Long-term BLER target enforcement via a feedback loop
        tau_inst = tau_inst + self.k_i * self.integral_error
        tau_inst = np.clip(tau_inst, .001, .99)

        # MCS selection via ILLA
        self.illa.bler_target = tau_inst
        mcs = self.illa(self.sinr_estimator.sinr)

        # Record history
        self.is_probe_hist.append(is_probe)
        self.tau_inst_hist.append(tau_inst)

        return mcs
