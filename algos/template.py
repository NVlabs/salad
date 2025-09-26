# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Link Adaptation algorithm template
"""

from abc import abstractmethod
import numpy as np

from utils import Sigmoid, SINREstimator


class ILLA:
    """
    Inner-Loop Link Adaptation (ILLA): Select the highest MCS index whose
    BLER is below the target

    Parameters:
    -----------

        bler_target: `float`
            BLER target

        bler_sigmoid_params: `dict`
            Sigmoid parameters approximating the BLER function

    Input:  
    ------

        sinr: `float` | `list` of `float`
            SINR values for which the MCS index is to be selected

    Output:
    -------

        mcs: `int`
            Selected MCS index

    """

    def __init__(self,
                 bler_target: float,
                 bler_sigmoid_params: dict):
        self.bler_target = bler_target
        self.n_mcs = len(bler_sigmoid_params['center'])
        self.likelihood_fun = Sigmoid(
            center=bler_sigmoid_params['center'][:, np.newaxis],
            scale=bler_sigmoid_params['scale'][:, np.newaxis])

    @property
    def bler_target(self):
        """ BLER target """
        return self._bler_target

    @bler_target.setter
    def bler_target(self,
                    value: float):
        assert 0 <= value <= 1, 'bler_target must be between 0 and 1'
        self._bler_target = value

    def __call__(self,
                 sinr,
                 **kwargs):

        if not hasattr(sinr, '__len__'):
            return_scalar = True
            sinr = np.array([sinr])
        else:
            return_scalar = False
            sinr = np.array(sinr)
        bler = 1 - self.likelihood_fun(sinr[np.newaxis, :])
        mcs = np.maximum((bler <= self.bler_target).sum(axis=0) - 1, 0)
        mcs = mcs[0] if return_scalar else mcs
        return mcs


class LinkAdaptation:
    """
    Link adaptation algorithm class template

    Parameters
    ----------

    bler_target : `float`
        Long-term BLER target

    bler_sigmoid_params : `dict`
        Sigmoid parameters approximating the BLER function

    sinr_estimator : `SINREstimator` | `None` (default)
        SINR estimator

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
                 bler_target : float,
                 bler_sigmoid_params : dict,
                 sinr_estimator : SINREstimator | None = None):
        if 'center' not in bler_sigmoid_params:
            raise ValueError('bler_sigmoid_params must contain a "center" key')
        if 'scale' not in bler_sigmoid_params:
            raise ValueError('bler_sigmoid_params must contain a "scale" key')
        if len(bler_sigmoid_params['center']) != len(bler_sigmoid_params['scale']):
            raise ValueError(
                'bler_sigmoid_params must contain the same number of "center" and "scale" values')
        if bler_target < 0 or bler_target > 1:
            raise ValueError('bler_target must be between 0 and 1')
        if sinr_estimator is not None:
            self.sinr_estimator = sinr_estimator
        self.bler_target = bler_target
        self.bler_sigmoid_params = bler_sigmoid_params

        # Unacknowledged MCS values
        self.mcs_unackned = []
        # Initialize ILLA
        self.illa = ILLA(self.bler_target,
                         self.bler_sigmoid_params)

    @abstractmethod
    def _mcs_selection(self):
        """
        MCS selection
        """

    def _process_harq_feedback(self,
                               is_nack):
        """
        Process HARQ feedback, e.g., update the SINR estimation
        """
        # Estimate the SINR
        self.sinr_estimator(is_nack)

    def _check_input_consistency(self,
                                 is_nack):
        """
        Check input consistency
        """
        # Check input consistency
        if len(is_nack) > len(self.mcs_unackned):
            raise ValueError(
                "len(is_nack) must not exceed len(self.mcs_unackned).\n"
                "You cannot receive HARQ feedback for packets that have not been transmitted yet.\n"
                f"len(is_nack) = {len(is_nack)}, len(self.mcs_unackned) = {len(self.mcs_unackned)}")

    def __call__(self,
                 is_nack=None,
                 **args):
        # Convert is_nack to list
        if is_nack is None:
            is_nack = []
        elif not hasattr(is_nack, '__len__'):
            is_nack = [is_nack]

        # Check input consistency
        self._check_input_consistency(is_nack)

        # Process HARQ feedback, e.g., update the SINR estimation
        self._process_harq_feedback(is_nack)

        # Select the MCS index
        mcs = self._mcs_selection()

        # Update unacknowledged MCS
        self.mcs_unackned = self.mcs_unackned[len(is_nack):] + [mcs]

        return mcs
