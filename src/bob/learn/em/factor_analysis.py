#!/usr/bin/env python
# @author: Tiago de Freitas Pereira


import functools
import logging
import operator

import dask
import dask.bag
import numpy as np

from dask.delayed import Delayed
from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels

from .gmm import GMMMachine
from .linear_scoring import linear_scoring
from .utils import check_and_persist_dask_input

logger = logging.getLogger(__name__)


def is_input_dask_nested(X):
    """Check if the input is a dask delayed or a (nested) list of dask delayed."""
    if isinstance(X, (list, tuple)):
        return is_input_dask_nested(X[0])

    if isinstance(X, Delayed):
        return True
    else:
        return False


def check_dask_input_samples_per_class(X, y):
    input_is_dask = is_input_dask_nested(X)

    if input_is_dask:
        y = functools.reduce(lambda x1, x2: list(x1) + list(x2), y)
    unique_labels_y = unique_labels(y)
    n_classes = len(unique_labels_y)
    n_samples_per_class = [sum(y == class_id) for class_id in unique_labels_y]
    return input_is_dask, n_classes, n_samples_per_class


def reduce_iadd(*args):
    """Reduces one or several lists by adding all elements into the first element"""
    ret = []
    for a in args:
        ret.append(functools.reduce(operator.iadd, a))

    if len(ret) == 1:
        return ret[0]
    return ret


def mult_along_axis(A, B, axis):
    """
    Magic function to multiply two arrays along a given axis.
    Taken from https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    """

    # shape check
    if axis >= A.ndim:
        raise np.AxisError(axis, A.ndim)
    if A.shape[axis] != B.size:
        raise ValueError(
            "Length of 'A' along the given axis must be the same as B.size"
        )

    # np.broadcast_to puts the new axis as the last axis, so
    # we swap the given axis with the last one, to determine the
    # corresponding array shape. np.swapaxes only returns a view
    # of the supplied array, so no data is copied unnecessarily.
    shape = np.swapaxes(A, A.ndim - 1, axis).shape

    # Broadcast to an array with the shape as above. Again,
    # no data is copied, we only get a new look at the existing data.
    B_brc = np.broadcast_to(B, shape)

    # Swap back the axes. As before, this only changes our "point of view".
    B_brc = np.swapaxes(B_brc, A.ndim - 1, axis)

    return A * B_brc


class FactorAnalysisBase(BaseEstimator):
    """
    Factor Analysis base class.
    This class is not intended to be used directly, but rather to be inherited from.
    For more information check [McCool2013]_ .


    Parameters
    ----------

    r_U: int
        Dimension of the subspace U

    r_V: int
        Dimension of the subspace V

    em_iterations: int
        Number of EM iterations

    relevance_factor: float
        Factor analysis relevance factor

    random_state: int
        random_state for the random number generator

    ubm: :py:class:`bob.learn.em.GMMMachine`
        A trained UBM (Universal Background Model) or a parametrized
        :py:class:`bob.learn.em.GMMMachine` to train the UBM with. If None,
        `ubm_kwargs` are passed as parameters of a new
        :py:class:`bob.learn.em.GMMMachine`.
    """

    def __init__(
        self,
        r_U,
        r_V=None,
        relevance_factor=4.0,
        em_iterations=10,
        random_state=0,
        enroll_iterations=1,
        ubm=None,
        ubm_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ubm = ubm
        self.ubm_kwargs = ubm_kwargs
        self.em_iterations = em_iterations
        self.random_state = random_state
        self.enroll_iterations = enroll_iterations

        # axis 1 dimensions of U and V
        self.r_U = r_U
        self.r_V = r_V

        self.relevance_factor = relevance_factor

        if ubm is not None and ubm._means is not None:
            self.create_UVD()

    @property
    def feature_dimension(self):
        """Get the UBM Dimension"""

        # TODO: Add this on the GMMMachine class
        return self.ubm.means.shape[1]

    @property
    def supervector_dimension(self):
        """Get the supervector dimension"""
        return self.ubm.n_gaussians * self.feature_dimension

    @property
    def mean_supervector(self):
        """
        Returns the mean supervector
        """
        return self.ubm.means.flatten()

    @property
    def variance_supervector(self):
        """
        Returns the variance supervector
        """
        return self.ubm.variances.flatten()

    @property
    def U(self):
        """An alias for `_U`."""
        return self._U

    @U.setter
    def U(self, value):
        self._U = np.array(value)

    @property
    def D(self):
        """An alias for `_D`."""
        return self._D

    @D.setter
    def D(self, value):
        self._D = np.array(value)

    @property
    def V(self):
        """An alias for `_V`."""
        return self._V

    @V.setter
    def V(self, value):
        self._V = np.array(value)

    def estimate_number_of_classes(self, y):
        """
        Estimates the number of classes given the labels
        """

        return len(unique_labels(y))

    def initialize_using_array(self, X):
        """
        Accumulating 0th and 1st order statistics. Trains the UBM if needed.

        Parameters
        ----------
        X: list of numpy arrays
            List of data to accumulate the statistics
        y: list of ints

        Returns
        -------

            n_acc: array
              (n_classes, n_gaussians) representing the accumulated 0th order statistics

            f_acc: array
                (n_classes, n_gaussians, feature_dim) representing the accumulated 1st order statistics

        """

        if self.ubm is None:
            logger.info("Creating a new GMMMachine and training it.")
            self.ubm = GMMMachine(**self.ubm_kwargs)
            self.ubm.fit(X)

        if self.ubm._means is None:
            logger.info("UBM means are None, training the UBM.")
            self.ubm.fit(X)

    def initialize(self, ubm_projected_X, y, n_classes):
        # Accumulating 0th and 1st order statistics
        # https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/ISVTrainer.cpp#L68

        logger.debug("Initializing the ISV/JFA using the UBM statistics.")

        # Initializing the state matrix
        if not hasattr(self, "_U") or not hasattr(self, "_D"):
            self.create_UVD()

        if is_input_dask_nested(ubm_projected_X):
            n_acc = [
                dask.delayed(self._sum_n_statistics)(xx, yy, n_classes)
                for xx, yy in zip(ubm_projected_X, y)
            ]

            f_acc = [
                dask.delayed(self._sum_f_statistics)(xx, yy, n_classes)
                for xx, yy in zip(ubm_projected_X, y)
            ]
            n_acc, f_acc = reduce_iadd(n_acc, f_acc)
        else:
            # 0th order stats
            n_acc = self._sum_n_statistics(ubm_projected_X, y, n_classes)
            # 1st order stats
            f_acc = self._sum_f_statistics(ubm_projected_X, y, n_classes)

        n_acc, f_acc = dask.compute(n_acc, f_acc)
        return n_acc, f_acc

    def create_UVD(self):
        """
        Create the state matrices U, V and D

        Returns
        -------

            U: (n_gaussians*feature_dimension, r_U) represents the session variability matrix (within-class variability)

            V: (n_gaussians*feature_dimension, r_V) represents the session variability matrix (between-class variability)

            D: (n_gaussians*feature_dimension) represents the client offset vector

        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        U_shape = (self.supervector_dimension, self.r_U)

        # U matrix is initialized using a normal distribution
        self._U = np.random.normal(scale=1.0, loc=0.0, size=U_shape)

        # D matrix is initialized as `D = sqrt(variance(UBM) / relevance_factor)`
        self._D = np.sqrt(self.variance_supervector / self.relevance_factor)

        # V matrix (or between-class variation matrix)
        # TODO: so far not doing JFA
        if self.r_V is not None:
            V_shape = (self.supervector_dimension, self.r_V)
            self._V = np.random.normal(scale=1.0, loc=0.0, size=V_shape)
        else:
            self._V = 0

    def _sum_n_statistics(self, X, y, n_classes):
        """
        Accumulates the 0th statistics for each client

        Parameters
        ----------
            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics of each sample

            y: list of ints
                List of corresponding labels

            n_classes: int
                Number of classes

        Returns
        -------
            n_acc: array
                (n_classes, n_gaussians) representing the accumulated 0th order statistics

        """
        # 0th order stats
        n_acc = np.zeros((n_classes, self.ubm.n_gaussians), like=X[0].n)

        # Iterate for each client
        for x_i, y_i in zip(X, y):
            # Accumulate the 0th statistics for each class
            n_acc[y_i, :] += x_i.n

        return n_acc

    def _sum_f_statistics(self, X, y, n_classes):
        """
        Accumulates the 1st order statistics for each client

        Parameters
        ----------
            X: list of :py:class:`bob.learn.em.GMMStats`

            y: list of ints
               List of corresponding labels

            n_classes: int
                Number of classes

        Returns
        -------
            f_acc: array
                (n_classes, n_gaussians, feature_dimension) representing the accumulated 1st order statistics

        """

        # 1st order stats
        f_acc = np.zeros(
            (
                n_classes,
                self.ubm.n_gaussians,
                self.feature_dimension,
            ),
            like=X[0].sum_px,
        )
        # Iterate for each client
        for x_i, y_i in zip(X, y):
            # Accumulate the 1st order statistics
            f_acc[y_i, :, :] += x_i.sum_px

        return f_acc

    def _get_statistics_by_class_id(self, X, y, i):
        """
        Returns the statistics for a given class

        Parameters
        ----------
            X: list of :py:class:`bob.learn.em.GMMStats`

            y: list of ints
                List of corresponding labels

            i: int
                Class id to return the statistics for
        """
        indices = np.where(np.array(y) == i)[0]
        return [X[i] for i in indices]

    """
    Estimating U and x
    """

    def _compute_id_plus_u_prod_ih(self, x_i, UProd):
        """
        Computes ( I+Ut*diag(sigma)^-1*Ni*U)^-1
        See equation (29) in [McCool2013]_

        Parameters
        ----------
            x_i: :py:class:`bob.learn.em.GMMStats`
                Statistics of a single sample

            UProd: array
                Matrix containing U_c.T*inv(Sigma_c) @ U_c.T

        Returns
        -------
            id_plus_u_prod_ih: array
                ( I+Ut*diag(sigma)^-1*Ni*U)^-1

        """

        n_i = x_i.n
        I = np.eye(self.r_U, self.r_U)  # noqa: E741

        # TODO: make the invertion matrix function as a parameter
        return np.linalg.inv(I + (UProd * n_i[:, None, None]).sum(axis=0))

    def _compute_fn_x_ih(self, x_i, latent_z_i=None, latent_y_i=None):
        """
        Computes Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i} - V*y_{i})
        Check equation (29) in [McCool2013]_

        Parameters
        ----------
            x_i: :py:class:`bob.learn.em.GMMStats`
                Statistics of a single sample

            latent_z_i: array
                E[z_i] for class `i`

            latent_y_i: array
                E[y_i] for class `i`

        """
        f_i = x_i.sum_px
        n_i = x_i.n
        n_ic = np.repeat(n_i, self.feature_dimension)
        V = self._V

        # N_ih*( m + D*z)
        # z is zero when the computation flow comes from update_X
        if latent_z_i is None:
            # Fn_x_ih = N_{i,h}*(o_{i,h} - m)
            fn_x_ih = f_i.flatten() - n_ic * (self.mean_supervector)
        else:
            # code goes here when the computation flow comes from compute_accumulators
            # Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i})
            fn_x_ih = f_i.flatten() - n_ic * (
                self.mean_supervector + self._D * latent_z_i
            )

        """
        # JFA Part (eq 29)
        """
        V_dot_v = V @ latent_y_i if latent_y_i is not None else 0
        fn_x_ih -= n_ic * V_dot_v if latent_y_i is not None else 0

        return fn_x_ih

    def compute_latent_x(
        self, *, X, y, n_classes, UProd, latent_y=None, latent_z=None
    ):
        """
        Computes a new math:`E[x]` See equation (29) in [McCool2013]_


        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of ints
                List of corresponding labels

            UProd: array
                Matrix containing U_c.T*inv(Sigma_c) @ U_c.T

            latent_y: array
                E(y) latent variable

            latent_z: array
                E(z) latent variable

        Returns
        -------
            Returns the new latent_x
        """

        # U.T @ inv(Sigma) - See Eq(37)
        UTinvSigma = self._U.T / self.variance_supervector

        if is_input_dask_nested(X):
            latent_x = [
                dask.delayed(self._compute_latent_x_per_class)(
                    X_i=X_i,
                    UProd=UProd,
                    UTinvSigma=UTinvSigma,
                    latent_y_i=latent_y[y_i] if latent_y is not None else None,
                    latent_z_i=latent_z[y_i] if latent_z is not None else None,
                )
                for y_i, X_i in enumerate(X)
            ]
            latent_x = dask.compute(*latent_x)
        else:
            latent_x = [None] * n_classes
            for y_i in unique_labels(y):
                latent_x[y_i] = self._compute_latent_x_per_class(
                    X_i=self._get_statistics_by_class_id(X, y, y_i),
                    UProd=UProd,
                    UTinvSigma=UTinvSigma,
                    latent_y_i=latent_y[y_i] if latent_y is not None else None,
                    latent_z_i=latent_z[y_i] if latent_z is not None else None,
                )

        return latent_x

    def _compute_latent_x_per_class(
        self, *, X_i, UProd, UTinvSigma, latent_y_i, latent_z_i
    ):
        # For each sample
        latent_x_i = []
        for x_i in X_i:
            id_plus_prod_ih = self._compute_id_plus_u_prod_ih(x_i, UProd)

            fn_x_ih = self._compute_fn_x_ih(
                x_i, latent_z_i=latent_z_i, latent_y_i=latent_y_i
            )
            latent_x_i.append(id_plus_prod_ih @ (UTinvSigma @ fn_x_ih))
        # make sure latent_x_i stays a dask array by converting the list to a
        # dask array explicitly
        latent_x_i = np.vstack(latent_x_i)
        latent_x_i = np.swapaxes(latent_x_i, 0, 1)
        return latent_x_i

    def update_U(self, acc_U_A1, acc_U_A2):
        """
        Update rule for U

        Parameters
        ----------

            acc_U_A1: array
                Accumulated statistics for U_A1(n_gaussians, r_U, r_U)

            acc_U_A2: array
                Accumulated statistics for U_A2(n_gaussians* feature_dimension, r_U)

        """
        # Inverting A1 over the zero axis
        # https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        inv_A1 = np.linalg.inv(acc_U_A1)

        # Iterating over the gaussians to update U
        U_c = (
            acc_U_A2.reshape(
                self.ubm.n_gaussians, self.feature_dimension, self.r_U
            )
            @ inv_A1
        )
        self._U = U_c.reshape(
            self.ubm.n_gaussians * self.feature_dimension, self.r_U
        )
        return self._U

    def _compute_uprod(self):
        """
        Computes U_c.T*inv(Sigma_c) @ U_c.T


        https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/FABaseTrainer.cpp#L325
        """
        # UProd = (self.ubm.n_gaussians, self.r_U, self.r_U)

        Uc = self._U.reshape(
            (self.ubm.n_gaussians, self.feature_dimension, self.r_U)
        )
        UcT = Uc.transpose(0, 2, 1)

        sigma_c = self.ubm.variances[:, np.newaxis]
        UProd = (UcT / sigma_c) @ Uc

        return UProd

    def compute_accumulators_U(self, X, y, UProd, latent_x, latent_y, latent_z):
        """
        Computes the accumulators (A1 and A2) for the U matrix.
        This is useful for parallelization purposes.

        The accumulators are defined as

        :math:`A_1 = \\sum\\limits_{i=1}^{I}\\sum\\limits_{h=1}^{H}N_{i,h,c}E(x_{i,h,c} x^{\\top}_{i,h,c})`


        :math:`A_2 = \\sum\\limits_{i=1}^{I}\\sum\\limits_{h=1}^{H}N_{i,h,c}(o_{i,h} - \\mu_c -D_{c}z_{i,c} -V_{c}y_{i,c} )E[x_{i,h}]^{\\top}`


        More information, please, check the technical notes attached

        Parameters
        ----------
            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of ints
                List of corresponding labels

            UProd: array
                Matrix containing U_c.T*inv(Sigma_c) @ U_c.T

            latent_x: array
                E(x) latent variable

            latent_y: array
                E(y) latent variable

            latent_z: array
                E(z) latent variable

        Returns
        -------
            acc_U_A1:
                (n_gaussians, r_U, r_U) A1 accumulator

            acc_U_A2:
                (n_gaussians* feature_dimension, r_U) A2 accumulator


        """

        # U accumulators
        acc_U_A1 = np.zeros(
            (self.ubm.n_gaussians, self.r_U, self.r_U), like=X[0].n
        )
        acc_U_A2 = np.zeros((self.supervector_dimension, self.r_U), like=X[0].n)

        # Loops over all people
        for y_i in set(y):
            # For each session
            for session_index, x_i in enumerate(
                self._get_statistics_by_class_id(X, y, y_i)
            ):
                id_plus_prod_ih = self._compute_id_plus_u_prod_ih(x_i, UProd)
                latent_z_i = latent_z[y_i] if latent_z is not None else None
                latent_y_i = latent_y[y_i] if latent_y is not None else None
                fn_x_ih = self._compute_fn_x_ih(
                    x_i, latent_y_i=latent_y_i, latent_z_i=latent_z_i
                )

                latent_x_i = latent_x[y_i][:, session_index]
                id_plus_prod_ih += (
                    latent_x_i[:, np.newaxis] @ latent_x_i[:, np.newaxis].T
                )

                acc_U_A1 += mult_along_axis(
                    id_plus_prod_ih[np.newaxis].repeat(
                        self.ubm.n_gaussians, axis=0
                    ),
                    x_i.n,
                    axis=0,
                )

                acc_U_A2 += fn_x_ih[np.newaxis].T @ latent_x_i[:, np.newaxis].T

        return acc_U_A1, acc_U_A2

    """
    Estimating D and z
    """

    def update_z(self, X, y, latent_x, latent_y, latent_z, n_acc, f_acc):
        """
        Computes a new math:`E[z]` See equation (30) in [McCool2013]_

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of ints
                List of corresponding labels

            latent_x: array
                E(x) latent variable

            latent_y: array
                E(y) latent variable

            latent_z: array
                E(z) latent variable

            n_acc: array
                Accumulated 0th order statistics for each class (math:`N_{i}`)

            f_acc: array
                Accumulated 1st order statistics for each class (math:`F_{i}`)

        Returns
        -------
            Returns the new latent_z

        """

        # Precomputing
        # self._D.T / sigma
        dt_inv_sigma = self._D / self.variance_supervector
        # self._D.T / sigma * self._D
        dt_inv_sigma_d = dt_inv_sigma * self._D

        # for each class
        for y_i in set(y):

            id_plus_d_prod = self._compute_id_plus_d_prod_i(
                dt_inv_sigma_d, n_acc[y_i]
            )
            X_i = self._get_statistics_by_class_id(X, y, y_i)
            latent_x_i = latent_x[y_i]

            latent_y_i = latent_y[y_i] if latent_y is not None else None

            fn_z_i = self._compute_fn_z_i(
                X_i, latent_x_i, latent_y_i, n_acc[y_i], f_acc[y_i]
            )
            latent_z[y_i] = id_plus_d_prod * dt_inv_sigma * fn_z_i

        return latent_z

    def _compute_id_plus_d_prod_i(self, dt_inv_sigma_d, n_acc_i):
        """
        Computes: (I+Dt*diag(sigma)^-1*Ni*D)^-1
        See equation (31) in [McCool2013]_

        Parameters
        ----------

         i: int
            Class id

        dt_inv_sigma_d: array
           Matrix representing `D.T / sigma`

        """

        tmp_CD = np.repeat(n_acc_i, self.feature_dimension)
        id_plus_d_prod = np.ones(tmp_CD.shape) + dt_inv_sigma_d * tmp_CD
        return 1 / id_plus_d_prod

    def _compute_fn_z_i(self, X_i, latent_x_i, latent_y_i, n_acc_i, f_acc_i):
        """
        Compute Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h}) (Normalized first order statistics)

        Parameters
        ----------
            i: int
                Class id

        """

        U = self._U
        V = self._V

        m = self.mean_supervector

        tmp_CD = np.repeat(n_acc_i, self.feature_dimension)

        # JFA session part
        V_dot_v = V @ latent_y_i if latent_y_i is not None else 0

        # m_cache_Fn_z_i = Fi - m_tmp_CD * (m + m_tmp_CD_b); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i})
        fn_z_i = f_acc_i.flatten() - tmp_CD * (m + V_dot_v)
        # convert fn_z_i to dask array here if required to make sure fn_z_i -= ... works
        fn_z_i = np.array(fn_z_i, like=X_i[0].n)

        # Looping over the sessions
        for session_id in range(len(X_i)):
            n_i = X_i[session_id].n
            tmp_CD = np.repeat(n_i, self.feature_dimension)
            x_i_h = latent_x_i[:, session_id]

            fn_z_i -= tmp_CD * (U @ x_i_h)

        return fn_z_i

    def compute_accumulators_D(
        self, X, y, latent_x, latent_y, latent_z, n_acc, f_acc
    ):
        """
        Compute the accumulators for the D matrix

        The accumulators are defined as

        :math:`A_1 = \\sum\\limits_{i=1}^{I}E[z_{i,c}z^{\\top}_{i,c}]`


        :math:`A_2 = \\sum\\limits_{i=1}^{I} \\Bigg[\\sum\\limits_{h=1}^{H}N_{i,h,c}(o_{i,h} - \\mu_c -U_{c}x_{i,h,c} -V_{c}y_{i,c} )\\Bigg]E[z_{i}]^{\\top}`


        More information, please, check the technical notes attached


        Parameters
        ----------

        X: array
            Input data

        y: array
            Class labels

        latent_z: array
            E(z)  latent variable

        latent_x: array
            E(x) latent variable

        latent_y: array
            E(y) latent variable

        n_acc: array
            Accumulated 0th order statistics for each class (math:`N_{i}`)

        f_acc: array
            Accumulated 1st order statistics for each class (math:`F_{i}`)

        Returns
        -------
            acc_D_A1:
                (n_gaussians* feature_dimension) A1 accumulator

            acc_D_A2:
                (n_gaussians* feature_dimension) A2 accumulator

        """

        acc_D_A1 = np.zeros((self.supervector_dimension,), like=X[0].n)
        acc_D_A2 = np.zeros((self.supervector_dimension,), like=X[0].n)

        # Precomputing
        # self._D.T / sigma
        dt_inv_sigma = self._D / self.variance_supervector
        # self._D.T / sigma * self._D
        dt_inv_sigma_d = dt_inv_sigma * self._D

        # Loops over all people
        for y_i in set(y):

            id_plus_d_prod = self._compute_id_plus_d_prod_i(
                dt_inv_sigma_d, n_acc[y_i]
            )
            X_i = self._get_statistics_by_class_id(X, y, y_i)
            latent_x_i = latent_x[y_i]

            latent_y_i = latent_y[y_i] if latent_y is not None else None

            fn_z_i = self._compute_fn_z_i(
                X_i, latent_x_i, latent_y_i, n_acc[y_i], f_acc[y_i]
            )

            tmp_CD = np.repeat(n_acc[y_i], self.feature_dimension)
            acc_D_A1 += (
                id_plus_d_prod + latent_z[y_i] * latent_z[y_i]
            ) * tmp_CD
            acc_D_A2 += fn_z_i * latent_z[y_i]

        return acc_D_A1, acc_D_A2

    def initialize_XYZ(self, n_samples_per_class, like=None):
        """
        Initialize E[x], E[y], E[z] state variables

        Eq. (38)
        latent_z = (n_classes, supervector_dimension)


        Eq. (37)
        latent_y = (n_classes, r_V) or None

        Eq. (36)
        latent_x = (n_classes, r_U, n_sessions)

        """
        kw = dict(like=like) if isinstance(like, dask.array.core.Array) else {}

        # x (Eq. 36)
        # (n_classes, r_U,  n_samples )
        latent_x = []
        for n_s in n_samples_per_class:
            latent_x.append(np.zeros((self.r_U, n_s), **kw))

        n_classes = len(n_samples_per_class)
        latent_y = (
            np.zeros((n_classes, self.r_V), **kw)
            if self.r_V and self.r_V > 0
            else None
        )

        latent_z = np.zeros((n_classes, self.supervector_dimension), **kw)

        return latent_x, latent_y, latent_z

    """
     Estimating V and y
    """

    def update_y(
        self,
        *,
        X,
        y,
        n_classes,
        VProd,
        latent_x,
        latent_y,
        latent_z,
        n_acc,
        f_acc,
    ):
        """
        Computes a new math:`E[y]` See equation (30) in [McCool2013]_

        Parameters
        ----------

        X: list of :py:class:`bob.learn.em.GMMStats`
            List of statistics

        y: list of ints
            List of corresponding labels

        n_classes: int
            Number of classes

        VProd: array
            Matrix representing V_c.T*inv(Sigma_c) @ V_c.T

        latent_x: array
            E(x) latent variable

        latent_y: array
            E(y) latent variable

        latent_z: array
            E(z) latent variable

        n_acc: array
            Accumulated 0th order statistics for each class (math:`N_{i}`)

        f_acc: array
            Accumulated 1st order statistics for each class (math:`F_{i}`)

        """
        # V.T / sigma
        VTinvSigma = self._V.T / self.variance_supervector

        if is_input_dask_nested(X):
            latent_y = [
                dask.delayed(self._latent_y_per_class)(
                    X_i=X_i,
                    n_acc_i=n_acc[label],
                    f_acc_i=f_acc[label],
                    VProd=VProd,
                    VTinvSigma=VTinvSigma,
                    latent_x_i=latent_x[label],
                    latent_z_i=latent_z[label],
                )
                for label, X_i in enumerate(X)
            ]
            latent_y = dask.compute(*latent_y)
        else:
            # Loops over the labels
            for label in range(n_classes):
                X_i = self._get_statistics_by_class_id(X, y, label)
                latent_y[label] = self._latent_y_per_class(
                    X_i=X_i,
                    n_acc_i=n_acc[label],
                    f_acc_i=f_acc[label],
                    VProd=VProd,
                    VTinvSigma=VTinvSigma,
                    latent_x_i=latent_x[label],
                    latent_z_i=latent_z[label],
                )
        return latent_y

    def _latent_y_per_class(
        self,
        *,
        X_i,
        n_acc_i,
        f_acc_i,
        VProd,
        VTinvSigma,
        latent_x_i,
        latent_z_i,
    ):
        id_plus_v_prod_i = self._compute_id_plus_vprod_i(n_acc_i, VProd)
        fn_y_i = self._compute_fn_y_i(
            X_i,
            latent_x_i,
            latent_z_i,
            n_acc_i,
            f_acc_i,
        )
        latent_y = (VTinvSigma @ fn_y_i) @ id_plus_v_prod_i
        return latent_y

    def _compute_id_plus_vprod_i(self, n_acc_i, VProd):
        """
        Computes: (I+Vt*diag(sigma)^-1*Ni*V)^-1 (see Eq. (30) in [McCool2013]_)

        Parameters
        ----------

        n_acc_i: array
            Accumulated 0th order statistics for each class (math:`N_{i}`)

        VProd: array
            Matrix representing V_c.T*inv(Sigma_c) @ V_c.T

        """

        I = np.eye(self.r_V, self.r_V)  # noqa: E741

        # TODO: make the invertion matrix function as a parameter
        return np.linalg.inv(I + (VProd * n_acc_i[:, None, None]).sum(axis=0))

    def _compute_vprod(self):
        """
        Computes V_c.T*inv(Sigma_c) @ V_c.T


        https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/FABaseTrainer.cpp#L193
        """

        Vc = self._V.reshape(
            (self.ubm.n_gaussians, self.feature_dimension, self.r_V)
        )
        VcT = Vc.transpose(0, 2, 1)

        sigma_c = self.ubm.variances[:, np.newaxis]
        VProd = (VcT / sigma_c) @ Vc

        return VProd

    def compute_accumulators_V(
        self, X, y, VProd, n_acc, f_acc, latent_x, latent_y, latent_z
    ):
        """
        Computes the accumulators for the update of V matrix
        The accumulators are defined as

        :math:`A_1 = \\sum\\limits_{i=1}^{I}E[y_{i,c}y^{\\top}_{i,c}]`


        :math:`A_2 = \\sum\\limits_{i=1}^{I} \\Bigg[\\sum\\limits_{h=1}^{H}N_{i,h,c}(o_{i,h} - \\mu_c -U_{c}x_{i,h,c} -D_{c}z_{i,c} )\\Bigg]E[y_{i}]^{\\top}`


        More information, please, check the technical notes attached

        Parameters
        ----------

        X: list of :py:class:`bob.learn.em.GMMStats`
            List of statistics

        y: list of ints
            List of corresponding labels

        VProd: array
            Matrix representing V_c.T*inv(Sigma_c) @ V_c.T

        n_acc: array
            Accumulated 0th order statistics for each class (math:`N_{i}`)

        f_acc: array
            Accumulated 1st order statistics for each class (math:`F_{i}`)

        latent_x: array
            E(x) latent variable

        latent_y: array
            E(y) latent variable

        latent_z: array
            E(z) latent variable


        Returns
        -------

            acc_V_A1:
                (n_gaussians, r_V, r_V) A1 accumulator

            acc_V_A2:
                (n_gaussians* feature_dimension, r_V) A2 accumulator

        """

        # U accumulators
        acc_V_A1 = np.zeros(
            (self.ubm.n_gaussians, self.r_V, self.r_V), like=X[0].n
        )
        acc_V_A2 = np.zeros((self.supervector_dimension, self.r_V), like=X[0].n)

        # Loops over all people
        for i in set(y):
            n_acc_i = n_acc[i]
            f_acc_i = f_acc[i]
            X_i = self._get_statistics_by_class_id(X, y, i)
            latent_x_i = latent_x[i]
            latent_y_i = latent_y[i]
            latent_z_i = latent_z[i]

            # Computing A1 accumulator: \sum_{i=1}^{N}(E(y_i_c @ y_i_c.T))
            id_plus_prod_v_i = self._compute_id_plus_vprod_i(n_acc_i, VProd)
            id_plus_prod_v_i += (
                latent_y_i[:, np.newaxis] @ latent_y_i[:, np.newaxis].T
            )

            acc_V_A1 += mult_along_axis(
                id_plus_prod_v_i[np.newaxis].repeat(
                    self.ubm.n_gaussians, axis=0
                ),
                n_acc_i,
                axis=0,
            )

            # Computing A2 accumulator: \sum_{i=1}^{N}( \sum_{h=1}^{H}(N_i_h_c (o_i_h, - m_c - D_c*z_i_c - U_c*x_i_h_c))@ E(y_i).T   )
            fn_y_i = self._compute_fn_y_i(
                X_i,
                latent_x_i=latent_x_i,
                latent_z_i=latent_z_i,
                n_acc_i=n_acc_i,
                f_acc_i=f_acc_i,
            )

            acc_V_A2 += fn_y_i[np.newaxis].T @ latent_y_i[:, np.newaxis].T

        return acc_V_A1, acc_V_A2

    def _compute_fn_y_i(self, X_i, latent_x_i, latent_z_i, n_acc_i, f_acc_i):
        """
        // Compute Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) (Normalized first order statistics)
        See equation (30) in [McCool2013]_

        Parameters
        ----------

        X_i: list of :py:class:`bob.learn.em.GMMStats`
            List of statistics for a class

        latent_x_i: array
            E(x_i) latent variable

        latent_z_i: array
            E(z_i) latent variable

        n_acc_i: array
            Accumulated 0th order statistics for each class (math:`N_{i}`)

        f_acc_i: array
            Accumulated 1st order statistics for each class (math:`F_{i}`)


        """

        U = self._U
        D = self._D  # Not doing the JFA

        m = self.mean_supervector

        # y = self.y[i] # Not doing JFA

        tmp_CD = np.repeat(n_acc_i, self.feature_dimension)

        fn_y_i = f_acc_i.flatten() - tmp_CD * (
            m - D * latent_z_i
        )  # Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i})

        # Looping over the sessions of a ;ane;
        for session_id in range(len(X_i)):
            n_i = X_i[session_id].n
            U_dot_x = U @ latent_x_i[:, session_id]
            tmp_CD = np.repeat(n_i, self.feature_dimension)
            fn_y_i -= tmp_CD * U_dot_x

        return fn_y_i

    """
     Scoring
    """

    def estimate_x(self, X):

        id_plus_us_prod_inv = self._compute_id_plus_us_prod_inv(X)
        fn_x = self._compute_fn_x(X)

        # UtSigmaInv * Fn_x = Ut*diag(sigma)^-1 * N*(o - m)
        ut_inv_sigma = self._U.T / self.variance_supervector

        return id_plus_us_prod_inv @ (ut_inv_sigma @ fn_x)

    def estimate_ux(self, X):
        x = self.estimate_x(X)
        return self.U @ x

    def _compute_id_plus_us_prod_inv(self, X_i):
        """
        Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 =

        Parameters
        ----------

            X_i: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics for a class
        """
        I = np.eye(self.r_U, self.r_U)  # noqa: E741

        Uc = self._U.reshape(
            (self.ubm.n_gaussians, self.feature_dimension, self.r_U)
        )

        UcT = np.transpose(Uc, axes=(0, 2, 1))

        sigma_c = self.ubm.variances[:, np.newaxis]

        n_i_c = np.expand_dims(X_i.n[:, np.newaxis], axis=2)

        id_plus_us_prod_inv = I + (((UcT / sigma_c) @ Uc) * n_i_c).sum(axis=0)
        id_plus_us_prod_inv = np.linalg.inv(id_plus_us_prod_inv)

        return id_plus_us_prod_inv

    def _compute_fn_x(self, X_i):
        """
        Compute Fn_x = sum_{sessions h}(N*(o - m) (Normalized first order statistics)

            Parameters
            ----------

            X_i: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics for a class

        """

        n = X_i.n[:, np.newaxis]
        f = X_i.sum_px

        fn_x = f - self.ubm.means * n

        return fn_x.flatten()

    def score_using_array(self, model, data):
        """
        Computes the ISV score using a numpy array as input

        Parameters
        ----------
        latent_z : numpy.ndarray
            Latent representation of the client (E[z_i])

        data : list of :py:class:`bob.learn.em.GMMStats`
            List of statistics to be scored

        Returns
        -------
        score : float
            The linear scored

        """

        return self.score(model, self.ubm.acc_stats(data))

    def fit_using_array(self, X, y):
        """Fits the model using a numpy array or a dask array as input
        The y matrix is computed to a numpy array immediately.
        """

        input_is_dask, X = check_and_persist_dask_input(X, persist=True)
        y = dask.compute(y)[0]
        y = np.squeeze(np.asarray(y))
        check_consistent_length(X, y)

        self.initialize_using_array(X)

        if input_is_dask:
            # split the X array based on the classes
            # since the EM algorithm is only parallelizable per class
            X_new, y_new = [], []
            for class_id in unique_labels(y):
                class_indices = y == class_id
                X_new.append(X[class_indices])
                y_new.append(y[class_indices])
            X, y = X_new, y_new
            del X_new, y_new

            stats = [dask.delayed(self.ubm.transform)(xx).persist() for xx in X]
        else:
            stats = self.ubm.transform(X)

        logger.info("Computing statistics per sample")
        del X
        self.fit(stats, y)
        return self

    def enroll_using_array(self, X):
        """
        Enrolls a new client using a numpy array as input

        Parameters
        ----------
        X : array
            features to be enrolled

        iterations : int
            Number of iterations to perform

        Returns
        -------
        self : object
            z

        """
        return self.enroll([self.ubm.acc_stats(X)])

    def _prepare_dask_input(self, X, y):
        """Perpare the input for the fit method"""
        logger.info(
            "Rechunking bag of stats to delayed list of stats per class. If your worker runs "
            "out of memory in this training step, you have to use workers with more memory."
        )
        # optimize the graph of X and y and also persist X before computing y
        X, y = dask.optimize(X, y)
        X = X.persist()
        y = np.array(dask.compute(y)[0])
        n_classes = len(set(y))

        # X is a list Stats in a dask bag chunked randomly. We want X to be a
        # list of dask delayed objects where each delayed object is a list of
        # stats per class
        def _len(stats):
            return len(stats)

        lengths = X.map_partitions(_len).compute()
        delayeds = X.to_delayed()
        X, i = [[] for _ in range(n_classes)], 0
        for length_, delayed_stats_list in zip(lengths, delayeds):
            delayed_stats_list._length = length_
            for delayed_stat in delayed_stats_list:
                class_id = y[i]
                X[class_id].append(delayed_stat)
                i += 1
        X = [dask.delayed(list)(stats).persist() for stats in X]
        y = [y[y == class_id] for class_id in range(n_classes)]

        return X, y


class ISVMachine(FactorAnalysisBase):
    """
    Implements the Intersession Variability Modelling hypothesis on top of GMMs

    Inter-Session Variability (ISV) modeling is a session variability modeling technique built on top of the Gaussian mixture modeling approach.
    It hypothesizes that within-class variations are embedded in a linear subspace in the GMM means subspace and these variations can be suppressed
    by an offset w.r.t each mean during the MAP adaptation.
    For more information check [McCool2013]_

    Parameters
    ----------

    r_U: int
        Dimension of the subspace U

    em_iterations: int
        Number of EM iterations

    relevance_factor: float
        Factor analysis relevance factor

    random_state: int
        random_state for the random number generator

    ubm: :py:class:`bob.learn.em.GMMMachine` or None
        A trained UBM (Universal Background Model). If None, the UBM is trained with
        a new :py:class:`bob.learn.em.GMMMachine` when fit is called, with `ubm_kwargs`
        as parameters.

    """

    def __init__(
        self,
        r_U,
        em_iterations=10,
        relevance_factor=4.0,
        random_state=0,
        ubm=None,
        ubm_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            r_U=r_U,
            relevance_factor=relevance_factor,
            em_iterations=em_iterations,
            random_state=random_state,
            ubm=ubm,
            ubm_kwargs=ubm_kwargs,
            **kwargs,
        )

    def e_step(self, X, y, n_samples_per_class, n_acc, f_acc):
        """
        E-step of the EM algorithm
        """
        # self.initialize_XYZ(y)
        UProd = self._compute_uprod()
        _, _, latent_z = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class, like=X[0].n
        )
        latent_y = None

        latent_x = self.compute_latent_x(
            X=X,
            y=y,
            n_classes=len(n_samples_per_class),
            UProd=UProd,
        )
        latent_z = self.update_z(
            X=X,
            y=y,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
            n_acc=n_acc,
            f_acc=f_acc,
        )
        acc_U_A1, acc_U_A2 = self.compute_accumulators_U(
            X, y, UProd, latent_x, latent_y, latent_z
        )

        return acc_U_A1, acc_U_A2

    def m_step(self, acc_U_A1_acc_U_A2_list):
        """
        ISV M-step.
        This updates `U` matrix

        Parameters
        ----------

            acc_U_A1: array
                Accumulated statistics for U_A1(n_gaussians, r_U, r_U)

            acc_U_A2: array
                Accumulated statistics for U_A2(n_gaussians* feature_dimension, r_U)

        """
        acc_U_A1 = [acc[0] for acc in acc_U_A1_acc_U_A2_list]
        acc_U_A2 = [acc[1] for acc in acc_U_A1_acc_U_A2_list]

        acc_U_A1, acc_U_A2 = reduce_iadd(acc_U_A1, acc_U_A2)

        return self.update_U(acc_U_A1, acc_U_A2)

    def fit(self, X, y):
        """
        Trains the U matrix (session variability matrix)

        Parameters
        ----------
        X : numpy.ndarray
            Nxd features of N GMM statistics
        y : numpy.ndarray
            The input labels, a 1D numpy array of shape (number of samples, )

        Returns
        -------
        self : object
            Returns self.

        """
        if isinstance(X, dask.bag.Bag):
            X, y = self._prepare_dask_input(X, y)

        (
            input_is_dask,
            n_classes,
            n_samples_per_class,
        ) = check_dask_input_samples_per_class(X, y)

        n_acc, f_acc = self.initialize(X, y, n_classes)

        for i in range(self.em_iterations):
            logger.info("ISV U Training: Iteration %d", i + 1)
            if input_is_dask:
                e_step_output = [
                    dask.delayed(self.e_step)(
                        X=xx,
                        y=yy,
                        n_samples_per_class=n_samples_per_class,
                        n_acc=n_acc,
                        f_acc=f_acc,
                    )
                    for xx, yy in zip(X, y)
                ]
                delayed_em_step = dask.delayed(self.m_step)(e_step_output)
                self._U = dask.compute(delayed_em_step)[0]
            else:
                e_step_output = self.e_step(
                    X=X,
                    y=y,
                    n_samples_per_class=n_samples_per_class,
                    n_acc=n_acc,
                    f_acc=f_acc,
                )
                self.m_step([e_step_output])

        return self

    def transform(self, X):
        ubm_projected_X = self.ubm.acc_stats(X)
        return self.estimate_ux(ubm_projected_X)

    def enroll(self, X):
        """
        Enrolls a new client
        In ISV, the enrolment is defined as: :math:`m + Dz` with the latent variables `z`
        representing the enrolled model.

        Parameters
        ----------
        X : list of :py:class:`bob.learn.em.GMMStats`
            List of statistics to be enrolled

        Returns
        -------
        self : object
            z

        """
        iterations = self.enroll_iterations
        # We have only one class for enrollment
        y = list(np.zeros(len(X), dtype=np.int32))
        n_acc = self._sum_n_statistics(X, y=y, n_classes=1)
        f_acc = self._sum_f_statistics(X, y=y, n_classes=1)

        UProd = self._compute_uprod()
        _, _, latent_z = self.initialize_XYZ(n_samples_per_class=[len(X)])
        latent_y = None
        for i in range(iterations):
            logger.info("Enrollment: Iteration %d", i + 1)
            latent_x = self.compute_latent_x(
                X=X,
                y=y,
                n_classes=1,
                UProd=UProd,
                latent_y=latent_y,
                latent_z=latent_z,
            )
            latent_z = self.update_z(
                X=X,
                y=y,
                latent_x=latent_x,
                latent_y=latent_y,
                latent_z=latent_z,
                n_acc=n_acc,
                f_acc=f_acc,
            )

        return latent_z

    def enroll_using_array(self, X):
        """
        Enrolls a new client using a numpy array as input

        Parameters
        ----------
        X : array
            features to be enrolled

        iterations : int
            Number of iterations to perform

        Returns
        -------
        self : object
            z

        """
        return self.enroll([self.ubm.acc_stats(X)])

    def score(self, latent_z, data):
        """
        Computes the ISV score

        Parameters
        ----------
        latent_z : numpy.ndarray
            Latent representation of the client (E[z_i])

        data : list of :py:class:`bob.learn.em.GMMStats`
            List of statistics to be scored

        Returns
        -------
        score : float
            The linear scored

        """
        x = self.estimate_x(data)
        Ux = self._U @ x

        # TODO: I don't know why this is not the enrolled model
        # Here I am just reproducing the C++ implementation
        # m + Dz
        z = self.D * latent_z + self.mean_supervector

        return linear_scoring(
            z.reshape((self.ubm.n_gaussians, self.feature_dimension)),
            self.ubm,
            data,
            Ux.reshape((self.ubm.n_gaussians, self.feature_dimension)),
            frame_length_normalization=True,
        )[0]


class JFAMachine(FactorAnalysisBase):
    """
    Joint Factor Analysis (JFA) is an extension of ISV. Besides the
    within-class assumption (modeled with :math:`U`), it also hypothesize that
    between class variations are embedded in a low rank rectangular matrix
    :math:`V`. In the supervector notation, this modeling has the following shape:
    :math:`\\mu_{i, j} = m + Ux_{i, j}  + Vy_{i} + D_z{i}`.

    For more information check [McCool2013]_

    Parameters
    ----------

    ubm: :py:class:`bob.learn.em.GMMMachine`
        A trained UBM (Universal Background Model)

    r_U: int
        Dimension of the subspace U

    r_V: int
        Dimension of the subspace V

    em_iterations: int
        Number of EM iterations

    relevance_factor: float
        Factor analysis relevance factor

    random_state: int
        random_state for the random number generator

    """

    def __init__(
        self,
        r_U,
        r_V,
        em_iterations=10,
        relevance_factor=4.0,
        random_state=0,
        ubm=None,
        ubm_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            ubm=ubm,
            r_U=r_U,
            r_V=r_V,
            relevance_factor=relevance_factor,
            em_iterations=em_iterations,
            random_state=random_state,
            ubm_kwargs=ubm_kwargs,
            **kwargs,
        )

    def e_step_v(self, X, y, n_samples_per_class, n_acc, f_acc):
        """
        ISV E-step for the V matrix.

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of int
                List of labels

            n_classes: int
                Number of classes

            n_acc: array
                Accumulated 0th-order statistics

            f_acc: array
                Accumulated 1st-order statistics


        Returns
        ----------

            acc_V_A1: array
                Accumulated statistics for V_A1(n_gaussians, r_V, r_V)

            acc_V_A2: array
                Accumulated statistics for V_A2(n_gaussians* feature_dimension, r_V)

        """

        VProd = self._compute_vprod()

        latent_x, latent_y, latent_z = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class, like=X[0].n
        )

        # UPDATE Y, X AND FINALLY Z

        n_classes = len(n_samples_per_class)
        latent_y = self.update_y(
            X=X,
            y=y,
            n_classes=n_classes,
            VProd=VProd,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
            n_acc=n_acc,
            f_acc=f_acc,
        )

        acc_V_A1, acc_V_A2 = self.compute_accumulators_V(
            X=X,
            y=y,
            VProd=VProd,
            n_acc=n_acc,
            f_acc=f_acc,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
        )

        return acc_V_A1, acc_V_A2

    def m_step_v(self, acc_V_A1_acc_V_A2_list):
        """
        `V` Matrix M-step.
        This updates the `V` matrix

        Parameters
        ----------

            acc_V_A1: array
                Accumulated statistics for V_A1(n_gaussians, r_V, r_V)

            acc_V_A2: array
                Accumulated statistics for V_A2(n_gaussians* feature_dimension, r_V)

        """
        acc_V_A1 = [acc[0] for acc in acc_V_A1_acc_V_A2_list]
        acc_V_A2 = [acc[1] for acc in acc_V_A1_acc_V_A2_list]

        acc_V_A1, acc_V_A2 = reduce_iadd(acc_V_A1, acc_V_A2)

        # Inverting A1 over the zero axis
        # https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        inv_A1 = np.linalg.inv(acc_V_A1)

        V_c = (
            acc_V_A2.reshape(
                (self.ubm.n_gaussians, self.feature_dimension, self.r_V)
            )
            @ inv_A1
        )
        self._V = V_c.reshape(
            (self.ubm.n_gaussians * self.feature_dimension, self.r_V)
        )
        return self._V

    def finalize_v(self, X, y, n_samples_per_class, n_acc, f_acc):
        """
        Compute for the last time `E[y]`

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of int
                List of labels

            n_classes: int
                Number of classes

            n_acc: array
                Accumulated 0th-order statistics

            f_acc: array
                Accumulated 1st-order statistics

        Returns
        -------
            latent_y: array
                E[y]

        """
        VProd = self._compute_vprod()

        latent_x, latent_y, latent_z = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class, like=X[0].n
        )

        # UPDATE Y, X AND FINALLY Z

        n_classes = len(n_samples_per_class)
        latent_y = self.update_y(
            X=X,
            y=y,
            n_classes=n_classes,
            VProd=VProd,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
            n_acc=n_acc,
            f_acc=f_acc,
        )
        return latent_y

    def e_step_u(self, X, y, n_samples_per_class, latent_y):
        """
        ISV E-step for the U matrix.

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of int
                List of labels

            latent_y: array
                E(y) latent variable


        Returns
        ----------

            acc_U_A1: array
                Accumulated statistics for U_A1(n_gaussians, r_U, r_U)

            acc_U_A2: array
                Accumulated statistics for U_A2(n_gaussians* feature_dimension, r_U)

        """
        # self.initialize_XYZ(y)
        UProd = self._compute_uprod()
        latent_x, _, latent_z = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class
        )

        n_classes = len(n_samples_per_class)
        latent_x = self.compute_latent_x(
            X=X,
            y=y,
            n_classes=n_classes,
            UProd=UProd,
            latent_y=latent_y,
        )

        acc_U_A1, acc_U_A2 = self.compute_accumulators_U(
            X=X,
            y=y,
            UProd=UProd,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
        )

        return acc_U_A1, acc_U_A2

    def m_step_u(self, acc_U_A1_acc_U_A2_list):
        """
        `U` Matrix M-step.
        This updates the `U` matrix

        Parameters
        ----------

            acc_V_A1: array
                Accumulated statistics for V_A1(n_gaussians, r_V, r_V)

            acc_V_A2: array
                Accumulated statistics for V_A2(n_gaussians* feature_dimension, r_V)

        """
        acc_U_A1 = [acc[0] for acc in acc_U_A1_acc_U_A2_list]
        acc_U_A2 = [acc[1] for acc in acc_U_A1_acc_U_A2_list]

        acc_U_A1, acc_U_A2 = reduce_iadd(acc_U_A1, acc_U_A2)

        return self.update_U(acc_U_A1, acc_U_A2)

    def finalize_u(
        self,
        X,
        y,
        n_samples_per_class,
        latent_y,
    ):
        """
        Compute for the last time `E[x]`

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of int
                List of labels

            n_classes: int
                Number of classes

            latent_y: array
                E[y] latent variable

        Returns
        -------
            latent_x: array
                E[x]
        """

        UProd = self._compute_uprod()
        latent_x, _, _ = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class
        )

        n_classes = len(n_samples_per_class)
        latent_x = self.compute_latent_x(
            X=X,
            y=y,
            n_classes=n_classes,
            UProd=UProd,
            latent_y=latent_y,
        )

        return latent_x

    def e_step_d(
        self, X, y, n_samples_per_class, latent_x, latent_y, n_acc, f_acc
    ):
        """
        ISV E-step for the U matrix.

        Parameters
        ----------

            X: list of :py:class:`bob.learn.em.GMMStats`
                List of statistics

            y: list of int
                List of labels

            n_classes: int
                Number of classes

            latent_x: array
                E(x) latent variable

            latent_y: array
                E(y) latent variable

            latent_z: array
                E(z) latent variable

            n_acc: array
                Accumulated 0th-order statistics

            f_acc: array
                Accumulated 1st-order statistics


        Returns
        ----------

            acc_D_A1: array
                Accumulated statistics for D_A1(n_gaussians* feature_dimension, )

            acc_D_A2: array
                Accumulated statistics for D_A2(n_gaussians* feature_dimension, )

        """
        _, _, latent_z = self.initialize_XYZ(
            n_samples_per_class=n_samples_per_class
        )

        latent_z = self.update_z(
            X,
            y,
            latent_x=latent_x,
            latent_y=latent_y,
            latent_z=latent_z,
            n_acc=n_acc,
            f_acc=f_acc,
        )

        acc_D_A1, acc_D_A2 = self.compute_accumulators_D(
            X, y, latent_x, latent_y, latent_z, n_acc, f_acc
        )

        return acc_D_A1, acc_D_A2

    def m_step_d(self, acc_D_A1_acc_D_A2_list):
        """
        `D` Matrix M-step.
        This updates the `D` matrix

        Parameters
        ----------

            acc_D_A1: array
                Accumulated statistics for D_A1(n_gaussians* feature_dimension, )

            acc_D_A2: array
                Accumulated statistics for D_A2(n_gaussians* feature_dimension, )

        """
        acc_D_A1 = [acc[0] for acc in acc_D_A1_acc_D_A2_list]
        acc_D_A2 = [acc[1] for acc in acc_D_A1_acc_D_A2_list]

        acc_D_A1, acc_D_A2 = reduce_iadd(acc_D_A1, acc_D_A2)

        self._D = acc_D_A2 / acc_D_A1
        return self._D

    def enroll(self, X):
        """
        Enrolls a new client.
        In JFA the enrolment is defined as: :math:`m + Vy + Dz` with the latent variables `y` and `z`
        representing the enrolled model.

        Parameters
        ----------
        X : list of :py:class:`bob.learn.em.GMMStats`
            List of statistics

        Returns
        -------
        self : array
            z, y latent variables

        """
        iterations = self.enroll_iterations
        # We have only one class for enrollment
        y = list(np.zeros(len(X), dtype=np.int32))
        n_acc = self._sum_n_statistics(X, y=y, n_classes=1)
        f_acc = self._sum_f_statistics(X, y=y, n_classes=1)

        UProd = self._compute_uprod()
        VProd = self._compute_vprod()
        latent_x, latent_y, latent_z = self.initialize_XYZ(
            n_samples_per_class=[len(X)]
        )

        for i in range(iterations):
            logger.info("Enrollment: Iteration %d", i + 1)
            latent_y = self.update_y(
                X=X,
                y=y,
                n_classes=1,
                VProd=VProd,
                latent_x=latent_x,
                latent_y=latent_y,
                latent_z=latent_z,
                n_acc=n_acc,
                f_acc=f_acc,
            )
            latent_x = self.compute_latent_x(
                X=X,
                y=y,
                n_classes=1,
                UProd=UProd,
                latent_y=latent_y,
                latent_z=latent_z,
            )
            latent_z = self.update_z(
                X=X,
                y=y,
                latent_x=latent_x,
                latent_y=latent_y,
                latent_z=latent_z,
                n_acc=n_acc,
                f_acc=f_acc,
            )

        # The latent variables are wrapped in to 2axis arrays
        return latent_y[0], latent_z[0]

    def fit(self, X, y):
        """
        Trains the U matrix (session variability matrix)

        Parameters
        ----------
        X : numpy.ndarray
            Nxd features of N GMM statistics
        y : numpy.ndarray
            The input labels, a 1D numpy array of shape (number of samples, )

        Returns
        -------
        self : object
            Returns self.

        """
        if isinstance(X, dask.bag.Bag):
            X, y = self._prepare_dask_input(X, y)

        (
            input_is_dask,
            n_classes,
            n_samples_per_class,
        ) = check_dask_input_samples_per_class(X, y)

        n_acc, f_acc = self.initialize(X, y, n_classes=n_classes)

        # Updating V
        for i in range(self.em_iterations):
            logger.info("V Training: Iteration %d", i + 1)
            if input_is_dask:
                e_step_output = [
                    dask.delayed(self.e_step_v)(
                        X=xx,
                        y=yy,
                        n_samples_per_class=n_samples_per_class,
                        n_acc=n_acc,
                        f_acc=f_acc,
                    )
                    for xx, yy in zip(X, y)
                ]
                delayed_em_step = dask.delayed(self.m_step_v)(e_step_output)
                self._V = dask.compute(delayed_em_step)[0]
            else:
                e_step_output = self.e_step_v(
                    X=X,
                    y=y,
                    n_samples_per_class=n_samples_per_class,
                    n_acc=n_acc,
                    f_acc=f_acc,
                )
                self.m_step_v([e_step_output])
        latent_y = self.finalize_v(
            X=X,
            y=y,
            n_samples_per_class=n_samples_per_class,
            n_acc=n_acc,
            f_acc=f_acc,
        )

        # Updating U
        for i in range(self.em_iterations):
            logger.info("U Training: Iteration %d", i + 1)
            if input_is_dask:
                e_step_output = [
                    dask.delayed(self.e_step_u)(
                        X=xx,
                        y=yy,
                        n_samples_per_class=n_samples_per_class,
                        latent_y=latent_y,
                    )
                    for xx, yy in zip(X, y)
                ]
                delayed_em_step = dask.delayed(self.m_step_u)(e_step_output)
                self._U = dask.compute(delayed_em_step)[0]
            else:
                e_step_output = self.e_step_u(
                    X=X,
                    y=y,
                    n_samples_per_class=n_samples_per_class,
                    latent_y=latent_y,
                )
                self.m_step_u([e_step_output])

        latent_x = self.finalize_u(
            X=X, y=y, n_samples_per_class=n_samples_per_class, latent_y=latent_y
        )

        # Updating D
        for i in range(self.em_iterations):
            logger.info("D Training: Iteration %d", i + 1)
            if input_is_dask:
                e_step_output = [
                    dask.delayed(self.e_step_d)(
                        X=xx,
                        y=yy,
                        n_samples_per_class=n_samples_per_class,
                        latent_x=latent_x,
                        latent_y=latent_y,
                        n_acc=n_acc,
                        f_acc=f_acc,
                    )
                    for xx, yy in zip(X, y)
                ]
                delayed_em_step = dask.delayed(self.m_step_d)(e_step_output)
                self._D = dask.compute(delayed_em_step)[0]
            else:
                e_step_output = self.e_step_d(
                    X=X,
                    y=y,
                    n_samples_per_class=n_samples_per_class,
                    latent_x=latent_x,
                    latent_y=latent_y,
                    n_acc=n_acc,
                    f_acc=f_acc,
                )
                self.m_step_d([e_step_output])

        return self

    def score(self, model, data):
        """
        Computes the JFA score

        Parameters
        ----------
        latent_z : numpy.ndarray
            Latent representation of the client (E[z_i])

        data : list of :py:class:`bob.learn.em.GMMStats`
            List of statistics to be scored

        Returns
        -------
        score : float
            The linear scored

        """
        latent_y = model[0]
        latent_z = model[1]

        x = self.estimate_x(data)
        Ux = self._U @ x

        # TODO: I don't know why this is not the enrolled model
        # Here I am just reproducing the C++ implementation
        # m + Vy + Dz
        zy = self.V @ latent_y + self.D * latent_z + self.mean_supervector

        return linear_scoring(
            zy.reshape((self.ubm.n_gaussians, self.feature_dimension)),
            self.ubm,
            data,
            Ux.reshape((self.ubm.n_gaussians, self.feature_dimension)),
            frame_length_normalization=True,
        )[0]
