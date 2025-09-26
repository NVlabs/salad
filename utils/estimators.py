# © 2025 NVIDIA CORPORATION & AFFILIATES

"""
SINR estimator classes for SALAD and OLLA
"""

import numpy as np
from utils import BCELoss, Sigmoid, LagrangianInterpolator, Spline1stOrder


class SINREstimator:
    """
    Template class of a SINR estimator for link adaptation algorithms

    Input
    -----
        is_nack : `list` of `int` | `int`
            Whether HARQ feedback is NACK

    Output
    ------
        sinr : `float`
            SINR estimate for the current slot

    """

    def __init__(self):
        # Last SINR estimate
        self.sinr = None
        # History of SINR estimates
        self.sinr_hist = []

    def __call__(self,
                 is_nack,
                 **args):
        pass


class OLLASINREstimator(SINREstimator):
    """
    SINR estimator for outer-loop link adaptation (OLLA). The SINR offset is
    increased by `delta_nack` upon NACK and decreased by `delta_ack` upon
    ACK. `delta_ack` is computed as `delta_nack * bler_target / (1 - bler_target)`.

    Parameters
    ----------
        bler_target : `float`
            Long-term BLER target

        delta_nack : `float` (default: 1.)
            SINR offset adjustment upon NACK

        sinr_offset_init : `float` (default: 0.)
            Initial SINR offset

    Input
    -----
        is_nack : `list` of `int` | `int` (default: `None`)
            Whether HARQ feedback is NACK, if available

        sinr_report : `float` (default: `None`)
            External SINR estimate, e.g., from a CQI report, if available

    Output
    ------
        sinr : `float`
            SINR estimate for the current slot
    """

    def __init__(self,
                 bler_target,
                 delta_nack=1.,
                 sinr_offset_init=0.):
        super().__init__()
        # SINR offset adjustment upon NACK
        self.delta_nack = delta_nack
        # BLER target
        self.bler_target = bler_target
        # SINR offset adjustment upon ACK
        self.delta_ack = self.delta_nack * \
            self.bler_target / (1 - self.bler_target)
        # Initial SINR offset
        self.sinr_offset = sinr_offset_init
        # From last CQI report
        self.last_sinr_report = 0
        # Last SINR estimate
        self.sinr = self.sinr_offset + self.last_sinr_report
        self.sinr_hist = [self.sinr]

    def __call__(self,
                 is_nack=None,
                 sinr_report=None):
        if is_nack is None:
            is_nack = []
        elif not hasattr(is_nack, '__len__'):
            is_nack = [is_nack]
        for is_nack_i in is_nack:
            if is_nack_i not in [0, 1]:
                raise ValueError(f'is_nack must be 0 or 1, got {is_nack_i}')

        for is_nack_i in is_nack:
            if is_nack_i == 0:  # ACK
                self.sinr_offset += self.delta_ack
            else:  # NACK
                self.sinr_offset -= self.delta_nack

        if sinr_report is not None:
            # Consider new external estimation (CQI report)
            self.last_sinr_report = sinr_report

        # SINR offset and clip
        self.sinr = self.sinr_offset + self.last_sinr_report
        self.sinr_hist.append(self.sinr)
        return self.sinr


class TeacherSINREstimator(SINREstimator):
    """
    SALAD's teacher SINR estimator. The teacher minimizes the cross-entropy loss
    between the estimated BLER and the observed ACK/NACK feedback. 
    It uses a piece-wise linear interpolator with a number of knots that can be fixed or
    selected via cross-validation. 

    Parameters
    ----------

        bler_sigmoid_params : `dict`
            Sigmoid parameters approximating the BLER function

        n_knots : `int` (default: `None`)
            Number of knots. Only used if `cross_validation` is `False`

        cross_validation : `bool` (default: `True`)
            Whether to use cross-validation to select the number of knots

        training_portion : `float` (default: .5)
            Portion of the data to use for training. Only used if
            `cross_validation` is `True` 

        interp_type : 'linear' | 'lagrangian' (default: 'linear')
            Interpolator type.

        n_iter_gd : `int` (default: 1000)
            Number of gradient descent steps

        learning_rate : `float` (default: .001)
            Learning rate for gradient descent

        beta_regularization : `float` (default: 0)
            Regularization coefficient for temporal smoothness

        opt_over_last_n_samples : `int` (default: 300)
            Number of past samples to use for optimizing the model

        input_times : `bool` (default: `False`)
            Whether to input the timestamps with the feedback

        bler_clip_values : `tuple` of `float` (default: (.05, .95))
            BLER clip values

    Input
    -----
        is_nack : `list` of `int` | `int`
            Whether HARQ feedback is NACK

        mcs : `list` of `int` | `int`
            MCS index

        t : `list` of `float` | `float` (default: `None`)
            Slot index of MCS selection

        return_bler : `bool` (default: `False`)
            Whether to return the BLER estimate

    Output
    ------
        sinr : `float`
            SINR estimate for the current slot

        bler : `float`
            BLER estimate. Only returned if `return_bler` is `True`
    """

    def __init__(self,
                 bler_sigmoid_params,
                 n_knots=None,
                 cross_validation=False,
                 training_portion=.5,
                 interp_type='linear',
                 n_iter_gd=500,
                 learning_rate=.01,
                 beta_regularization=0,
                 opt_over_last_n_samples=300,
                 bler_clip_values=(.05, .95),
                 input_times=False,
                 n_knots_candidates=None):

        super().__init__()
        # Validate inputs
        if cross_validation and n_knots is not None:
            raise ValueError('n_knots must be None if cross_validation is True')
        if not cross_validation and n_knots is None:
            raise ValueError(
                'n_knots must be provided if cross_validation is False')
        if interp_type not in ['lagrangian', 'linear']:
            raise ValueError(f"Invalid interpolator type: {interp_type}. " +
                             "Must be one of 'lagrangian' or 'linear'")
        if interp_type == 'lagrangian':
            # Lagrangian interpolator
            interpolator = LagrangianInterpolator()
        else:
            # Piece-wise linear, i.e., 1st order spline, interpolator
            interpolator = Spline1stOrder()
        self.interp_type = interp_type
        self.interpolator = interpolator
        self.bler_sigmoid_params = bler_sigmoid_params
        self.n_knots = n_knots
        self.training_portion = training_portion
        self.n_iter_gd = n_iter_gd
        self.learning_rate = learning_rate
        self.beta_regularization = beta_regularization
        self.opt_over_last_n_samples = opt_over_last_n_samples
        self.input_times = input_times
        self.cross_validation = cross_validation
        self.bler_clip_values = bler_clip_values
        if n_knots_candidates is None:
            n_knots_candidates = np.r_[np.arange(2, 20, 2)]
        self.n_knots_candidates = n_knots_candidates
        self.sinr = 0

        # Initialize history
        self.sinr_hist = [self.sinr]
        self.is_nack_hist = []
        self.mcs_hist = []
        self.t_hist = []

        self.f_knots_star = None
        self.knots = None

    def reinit_history(self):
        """
        Reinitialize the history
        """
        self.is_nack_hist = []
        self.mcs_hist = []
        self.t_hist = []
        self.sinr_hist = []

    def _opt_fixed_model(self,
                         n_knots,
                         f_knots_init,
                         return_loss=False):
        """
        Optimize the model with fixed number of knots

        Inputs
        ------

            n_knots : `int`
                Number of knots

            f_knots_init : `np.ndarray`
                Initial knots

            return_loss : `bool` (default: `False`)
                Whether to return the loss values

        Outputs
        -------

            f_knots_star : `np.ndarray`
                Interpolated SINR estimation at knots

            knots : `np.ndarray`
                Knots

            loss_star : `float`
                Loss at the optimal knots. Only returned if `return_loss` is `True`

            loss_vec : `list`
                Loss values at each gradient descent step. Only returned if
                `return_loss` is `True` 

        """

        # Knot support is the same as for the observations
        knots = self.interpolator.get_knots(
            n_knots, [np.min(self.t_hist), np.max(self.t_hist)])

        # BLER function
        one_minus_bler_fun = Sigmoid(
            center=self.bler_sigmoid_params['center'][self.mcs_hist],
            scale=self.bler_sigmoid_params['scale'][self.mcs_hist])

        # Cross-entropy loss
        ce_loss = BCELoss(
            np.array(self.t_hist),
            1 - np.array(self.is_nack_hist),
            one_minus_bler_fun,
            self.interpolator,
            beta_regularization=self.beta_regularization)

        # Gradient descent
        f_knots_curr = f_knots_init
        f_knots_star = f_knots_init
        loss_star = ce_loss(knots, f_knots_star)
        if return_loss:
            loss_vec = [loss_star]
        for _ in range(self.n_iter_gd):
            # Compute gradient
            grad = ce_loss.gradient(knots, f_knots_curr)
            # Update f_knots
            f_knots_curr = f_knots_curr - grad * self.learning_rate
            # Compute loss
            loss = ce_loss(knots, f_knots_curr)
            if return_loss:
                loss_vec.append(loss)
            if loss < loss_star:
                # New best solution
                loss_star = loss
                f_knots_star = f_knots_curr

        if return_loss:
            return f_knots_star, knots, loss_star, loss_vec
        else:
            return f_knots_star, knots, loss_star

    def _opt_cross_validation(self):
        """
        Optimize the model by selecting the number of knots via cross-validation
        """
        n_obs = len(self.is_nack_hist)
        n_train = int(self.training_portion * n_obs)

        # Select training and test sets
        idx_perm = np.random.permutation(n_obs)
        idx_train = idx_perm[:n_train].astype(int)
        idx_test = idx_perm[n_train:].astype(int)

        t_train = np.array(self.t_hist)[idx_train]
        is_nack_train = np.array(self.is_nack_hist)[idx_train]
        mcs_train = np.array(self.mcs_hist)[idx_train]

        t_test = np.array(self.t_hist)[idx_test]
        is_nack_test = np.array(self.is_nack_hist)[idx_test]
        mcs_test = np.array(self.mcs_hist)[idx_test]

        # BLER function for test set
        one_minus_bler_test = Sigmoid(
            center=self.bler_sigmoid_params['center'][mcs_test],
            scale=self.bler_sigmoid_params['scale'][mcs_test])

        # Select the best number of knots
        loss_test_vec = []
        f_knots_star, knots_star = None, None
        loss_star = np.inf

        for n_knots in self.n_knots_candidates:

            # Instantiate teacher predictor with fixed number of knots
            fixed_n_knots_predictor = self.__class__(
                self.bler_sigmoid_params,
                n_knots=n_knots,
                cross_validation=False,
                interp_type=self.interp_type,
                n_iter_gd=self.n_iter_gd,
                learning_rate=self.learning_rate,
                beta_regularization=self.beta_regularization,
                opt_over_last_n_samples=self.opt_over_last_n_samples,
                input_times=True)

            # Train teacher predictor with fixed n. knots on training set
            fixed_n_knots_predictor(is_nack_train,
                                    mcs_train,
                                    t=t_train)
            knots_star_tmp = fixed_n_knots_predictor.knots
            f_knots_star_tmp = fixed_n_knots_predictor.f_knots_star

            # Compute the loss on test set
            ce_loss_test = BCELoss(
                t_test,
                1 - is_nack_test,
                one_minus_bler_test,
                self.interpolator)
            loss_test = ce_loss_test(knots_star_tmp,
                                     f_knots_star_tmp)
            loss_test_vec.append(loss_test)

            # Minimize the loss on test set
            # Select best solution so far
            if loss_test < loss_star:
                loss_star = loss_test
                f_knots_star = f_knots_star_tmp
                knots_star = knots_star_tmp

        return f_knots_star, knots_star, loss_star, loss_test_vec

    def __call__(self,
                 is_nack,
                 mcs,
                 t=None,
                 return_bler=False):
        # Validate inputs
        if t is None and self.input_times:
            raise ValueError('t must be provided if input_times is True')
        if t is not None and not self.input_times:
            raise ValueError('t must be None if input_times is False')

        if not hasattr(is_nack, '__len__'):
            is_nack = [is_nack]
        if not hasattr(mcs, '__len__'):
            mcs = [mcs]
        n_obs = len(is_nack)
        if n_obs != len(mcs):
            raise ValueError(f'is_nack and mcs must have the same length. '
                             f'Length of is_nack: {len(is_nack)}, length of mcs: {len(mcs)}')

        if t is not None:
            if not hasattr(t, '__len__'):
                t = [t]
            t = np.array(t)
            if len(t) != n_obs:
                raise ValueError('t and is_nack must have the same length')

        bler_vec = []

        if n_obs > 0:
            self.is_nack_hist.extend(is_nack)
            self.mcs_hist.extend(mcs)
            if self.input_times:
                self.t_hist.extend(t)
            else:
                self.t_hist = np.arange(len(self.is_nack_hist))

            # Consider last opt_over_last_n_samples samples
            self.is_nack_hist = self.is_nack_hist[-self.opt_over_last_n_samples:]
            self.mcs_hist = self.mcs_hist[-self.opt_over_last_n_samples:]
            self.t_hist = self.t_hist[-self.opt_over_last_n_samples:]

            # Initialize model: assign value to the knots
            if self.f_knots_star is None:
                # Random initialization
                f_knots_init = np.random.random(self.n_knots) * 2 - 1
            else:
                # Start from the last optimal solution
                f_knots_init = self.f_knots_star.copy()

            if not self.cross_validation:
                # Optimize the model with fixed number of knots
                self.f_knots_star, self.knots, *_ = self._opt_fixed_model(
                    self.n_knots,
                    f_knots_init)
            else:
                # Optimize the model by selecting the number of knots via
                # cross-validation
                self.f_knots_star, self.knots, *_ = self._opt_cross_validation()

            # Estimate SINR for every reported slot
            sinr_t_hist = self.interpolator(self.knots,
                                            self.f_knots_star,
                                            np.array(self.t_hist))
            # Return the last SINR estimation
            self.sinr = sinr_t_hist[-1]
            # Record history
            self.sinr_hist.extend(sinr_t_hist[-n_obs:])

            # Return BLER estimate
            if return_bler:
                sinr_vec = sinr_t_hist[-n_obs:]
                for mcs_i, sinr_i in zip(mcs, sinr_vec):

                    # BLER
                    one_minus_bler_fun = Sigmoid(
                        center=self.bler_sigmoid_params['center'][mcs_i],
                        scale=self.bler_sigmoid_params['scale'][mcs_i])
                    bler_i = 1 - one_minus_bler_fun(sinr_i)

                    # Clip BLER
                    bler_i = np.clip(bler_i,
                                     self.bler_clip_values[0],
                                     self.bler_clip_values[1])
                    bler_vec.append(bler_i)

        if not return_bler:
            return self.sinr
        else:
            return self.sinr, bler_vec


class StudentSINREstimator(SINREstimator):
    """
    SALAD's student SINR estimator. It estimates the next SINR value by gradient
    descent of the cross-entropy loss. 
    Its learning rate can be optimized via knowledge distillation from the
    teacher estimator. 

    Parameters
    ----------

        bler_sigmoid_params : `dict`
            Sigmoid parameters approximating the BLER function

        sinr_init : `float` (default: 10.)
            Initial SINR estimate

        learning_rate_init : `float` (default: 1.)
            Initial learning rate

        bler_clip_values : `tuple` of `float` (default: (.05, .95))
            BLER clip values. Useful to avoid numerical errors

        scale_clip_values : `tuple` of `float` (default: (0.5, 2))
            BLER's sigmoid scale clip values. Useful to avoid numerical errors

        teacher_estimator : `TeacherSINREstimator` (default: `None`)
            Teacher estimator. If `None`, knowledge distillation is not performed

        knowledge_distillation_period : `int` (default: `None`)
            Period for optimizing the learning rate via knowledge distillation.
            If `None`, knowledge distillation is not performed

    Input
    -----

        is_nack : `int`
            ACK/NACK feedback

        mcs : `int`
            MCS index

        return_bler : `bool` (default: `False`)
            Whether to return the estimated BLER

    Output
    ------

        sinr : `float`
            Estimated SINR for the current slot

        bler : `float`
            Estimated BLER. Only returned if `return_bler` is `True`
    """

    def __init__(self,
                 bler_sigmoid_params,
                 sinr_init=10.,
                 learning_rate_init=1.,
                 bler_clip_values=(.05, .95),
                 scale_clip_values=(.5, 2.),
                 teacher_estimator=None,
                 knowledge_distillation_period=None,
                 candidate_learning_rates=None):

        super().__init__()
        if 'center' not in bler_sigmoid_params:
            raise ValueError('bler_sigmoid_params must contain a "center" key')
        if 'scale' not in bler_sigmoid_params:
            raise ValueError('bler_sigmoid_params must contain a "scale" key')
        if len(bler_sigmoid_params['center']) != len(bler_sigmoid_params['scale']):
            raise ValueError(
                'bler_sigmoid_params must contain the same number of "center" and "scale" values')
        self.bler_sigmoid_params = bler_sigmoid_params
        self.learning_rate = learning_rate_init
        self.bler_clip_values = bler_clip_values
        self.scale_clip_values = scale_clip_values
        self.sinr_init = sinr_init
        self.sinr = sinr_init
        if knowledge_distillation_period is not None and teacher_estimator is None:
            raise ValueError(
                'teacher_estimator must be provided if knowledge_distillation_period is not None')
        if teacher_estimator is not None:
            if not isinstance(teacher_estimator, TeacherSINREstimator):
                raise ValueError(
                    'teacher_estimator must be an instance of TeacherSINREstimator')
        self.teacher_estimator = teacher_estimator
        if knowledge_distillation_period is None:
            # Knowledge distillation is not performed
            knowledge_distillation_period = np.inf
        self.knowledge_distillation_period = knowledge_distillation_period
        if candidate_learning_rates is None:
            candidate_learning_rates = np.arange(.1, 2, .1)
        self.candidate_learning_rates = candidate_learning_rates
        self.n_slots_seen = 0

        # Initialize history
        self.learning_rate_hist = [learning_rate_init]
        self.is_nack_hist = []
        self.mcs_hist = []
        self.sinr_hist = [sinr_init]

    @property
    def learning_rate(self):
        """ Get/set the learning rate of the gradient descent """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError('learning_rate must be positive')
        self._learning_rate = value

    def __call__(self,
                 is_nack,
                 mcs,
                 return_bler=False):
        # Transform inputs to lists
        if not hasattr(is_nack, '__len__'):
            is_nack = [is_nack]
        if not hasattr(mcs, '__len__'):
            mcs = [mcs]
        if len(is_nack) != len(mcs):
            raise ValueError(f'is_nack and mcs must have the same length. '
                             f'Length of is_nack: {len(is_nack)}, length of mcs: {len(mcs)}')

        # Update SINR estimate
        bler_vec = []
        for is_nack_i, mcs_i in zip(is_nack, mcs):
            self.n_slots_seen += 1

            # BLER
            one_minus_bler_fun = Sigmoid(
                center=self.bler_sigmoid_params['center'][mcs_i],
                scale=self.bler_sigmoid_params['scale'][mcs_i])
            bler_i = 1 - one_minus_bler_fun(self.sinr)

            # Clip BLER
            bler_i = np.clip(bler_i,
                             self.bler_clip_values[0],
                             self.bler_clip_values[1])
            bler_vec.append(bler_i)

            # Clip scale
            scale = np.clip(one_minus_bler_fun.scale,
                            self.scale_clip_values[0],
                            self.scale_clip_values[1])

            # Update the SINR estimation via gradient descent of the BCE loss
            increment = - self._get_d_bce_d_sinr(is_nack_i,
                                                 bler_i,
                                                 scale)

            self.sinr = self.sinr + self.learning_rate * increment

            # Record history
            self.sinr_hist.append(self.sinr)

            # Knowledge distillation for learning rate optimization
            if self.knowledge_distillation_period is not None:
                if self.n_slots_seen % self.knowledge_distillation_period == 0:
                    self.knowledge_distillation()

        self.is_nack_hist.extend(is_nack)
        self.mcs_hist.extend(mcs)
        if return_bler:
            return self.sinr, bler_vec
        return self.sinr

    def _get_d_bce_d_sinr(self, is_nack, bler, scale):
        """
        Compute the derivative of the BCE loss wrt the estimated SINR
        """

        # Derivative of loss wrt estimated SINR
        d_bce_d_sinr = (- (1 - is_nack) * bler + is_nack * (1 - bler)) / scale
        return d_bce_d_sinr

    def knowledge_distillation(self):
        """
        Optimize the student's learning rate via knowledge distillation. The
        teacher is trained on past samples. Then, multiple students are trained
        with different learning rates and the learning rate that minimizes the
        loss between the teacher and the student is selected

        Output
        ------

            new_learning_rate: float
                New learning rate

            sinr_teacher_hist: [sinr_teacher_histopt_over_last_n_samples]
                Teacher's past SINR estimation

            loss_teacher_student: [len(candidate_learning_rates)]
                Loss between teacher and student for every candidate learning rate
        """
        # Align MCS/NACK history
        is_nack_hist = self.is_nack_hist
        mcs_hist = self.mcs_hist[:len(is_nack_hist)]

        # Train teacher estimator
        self.teacher_estimator.reinit_history()
        self.teacher_estimator(is_nack_hist,
                               mcs_hist)
        sinr_teacher_hist = self.teacher_estimator.sinr_hist

        # Loss for every candidate student wrt teacher's SINR history
        loss_teacher_student = np.zeros(len(self.candidate_learning_rates))

        # Replay the student's SINR estimation for each candidate student with
        # different learning rates
        for ii, learning_rate in enumerate(self.candidate_learning_rates):
            student_estimator = self.__class__(
                self.bler_sigmoid_params,
                sinr_init=self.sinr_init,
                learning_rate_init=learning_rate)
            for mcs, ack in zip(mcs_hist, is_nack_hist):
                student_estimator(ack, mcs, return_bler=False)

            # Align student's SINR history with teacher's SINR history
            sinr_student_hist = np.array(
                student_estimator.sinr_hist[-len(sinr_teacher_hist):])

            # Compute loss between teacher and candidate student
            loss_teacher_student[ii] = np.mean(
                np.abs(sinr_student_hist - sinr_teacher_hist)**2)

        # Select learning rate that minimizes the loss between teacher and
        # candidate students for every learning rate
        new_learning_rate = self.candidate_learning_rates[np.argmin(
            loss_teacher_student)]

        # Set student's new learning rate
        self.learning_rate = new_learning_rate

        # Record history
        self.learning_rate_hist.append(new_learning_rate)

        return new_learning_rate, sinr_teacher_hist, loss_teacher_student
