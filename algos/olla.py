# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
Outer-Loop Link Adaptation (OLLA)
"""

from utils import OLLASINREstimator
from algos import LinkAdaptation


class OLLA(LinkAdaptation):
    """
    Outer-Loop Link Adaptation (OLLA)

    Parameters
    ----------

        bler_target: `float`
            BLER target

        bler_sigmoid_params: `dict`
            Sigmoid parameters approximating the BLER function

        sinr_estimator: `OLLASINREstimator` | `None` (default)
            SINR estimator

        delta_nack: `float`
            SINR offset adjustment upon NACK

        sinr_init: `float` (default: 0)
            Initial SINR estimate

    """

    def __init__(self,
                 bler_target: float,
                 bler_sigmoid_params: dict,
                 sinr_estimator : OLLASINREstimator):

        super().__init__(bler_target,
                         bler_sigmoid_params,
                         sinr_estimator=sinr_estimator)

    def _mcs_selection(self):
        """
        MCS selection via ILLA with respect to the SINR estimate
        """
        return self.illa(self.sinr_estimator.sinr)
