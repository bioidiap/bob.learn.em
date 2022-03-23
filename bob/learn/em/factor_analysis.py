#!/usr/bin/env python
# @author: Tiago de Freitas Pereira


import logging


import numpy as np
import scipy.spatial.distance

from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)
import bob.core


def mult_along_axis(A, B, axis):
    """
    Magic function to multiply two arrays along a given axis.
    Taken from https://stackoverflow.com/questions/30031828/multiply-numpy-ndarray-with-1d-array-along-a-given-axis
    """

    # ensure we're working with Numpy arrays
    A = np.array(A)
    B = np.array(B)

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
    GMM Factor Analysis base class

    """

    def __init__(self, ubm, r_U, r_V=None, relevance_factor=4.0):
        self.ubm = ubm

        # axis 1 dimensions of U and V
        self.r_U = r_U
        self.r_V = r_V

        self.relevance_factor = relevance_factor
        # Initializing the state matrix
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

    def estimate_number_of_classes(self, y):
        """
        Estimates the number of classes given the labels
        """

        return np.max(y) + 1

    def initialize(self, X, y):
        """
        Accumulating 0th and 1st order statistics

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

        # Accumulating 0th and 1st order statistics
        # https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/ISVTrainer.cpp#L68
        # 0th order stats
        n_acc = self._sum_n_statistics(X, y)

        # 1st order stats
        f_acc = self._sum_f_statistics(X, y)

        return n_acc, f_acc

    def create_UVD(self):
        """
        Create the state matrices U, V and D

        U: (n_gaussians*feature_dimension, r_U) represents the session variability matrix (within-class variability)

        V: (n_gaussians*feature_dimension, r_V) represents the session variability matrix (between-class variability)

        D: (n_gaussians*feature_dimension) represents the client offset vector

        """

        U_shape = (self.supervector_dimension, self.r_U)

        # U matrix is initialized using a normal distribution
        # https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/ISVTrainer.cpp#L72
        # TODO: Temporary workaround, so I can reuse the test cases
        if isinstance(self.seed, bob.core.random.mt19937):
            self.U = bob.core.random.variate_generator(
                bob.core.random.mt19937(0),
                bob.core.random.normal("float64", mean=0, sigma=1),
            )(shape=U_shape)
        else:
            # Assuming that the seed is an integer
            self.U = np.random.normal(scale=1.0, loc=0.0, size=U_shape)

        # D matrix is initialized as `D = sqrt(variance(UBM) / relevance_factor)`
        self.D = np.sqrt(self.variance_supervector / self.relevance_factor)

        # V matrix (or between-class variation matrix)
        # TODO: so far not doing JFA
        self.V = None

    def _get_statistics_by_class_id(self, X, y, i):
        """
        Returns the statistics for a given class
        """
        X = np.array(X)
        return list(X[np.where(np.array(y) == i)[0]])

    #################### Estimating U and x ######################

    def _computeUVD(self):
        """
        Precomputing `U.T @ inv(Sigma)`.
        See Eq 37

        TODO: I have to see if worth to keeping this cache

        """

        return self.U.T / self.variance_supervector

    def _sum_n_statistics(self, X, y):
        """
        Accumulates the 0th statistics for each client
        """
        # 0th order stats
        n_acc = np.zeros(
            (self.estimate_number_of_classes(y), self.ubm.n_gaussians)
        )

        # Iterate for each client
        for x_i, y_i in zip(X, y):
            # Accumulate the 0th statistics for each class
            n_acc[y_i, :] += x_i.n

        return n_acc

    def _sum_f_statistics(self, X, y):
        """
        Accumulates the 1st order statistics for each client
        """

        # 1st order stats
        f_acc = np.zeros(
            (
                self.estimate_number_of_classes(y),
                self.ubm.n_gaussians,
                self.feature_dimension,
            )
        )
        # Iterate for each client
        for x_i, y_i in zip(X, y):
            # Accumulate the 1st order statistics
            f_acc[y_i, :, :] += x_i.sum_px

        return f_acc

    def _compute_id_plus_prod_ih(self, x_i, y_i, UProd):
        """
        Computes ( I+Ut*diag(sigma)^-1*Ni*U)^-1)
        """

        n_i = x_i.n
        I = np.eye(self.r_U, self.r_U)

        # TODO: make the invertion matrix function as a parameter
        return np.linalg.inv(I + (UProd * n_i[:, None, None]).sum(axis=0))

    def _computefn_x_ih(self, x_i, y_i, latent_z=None):
        """
        Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i})
        """

        f_i = x_i.sum_px
        n_i = x_i.n
        n_ic = np.repeat(n_i, self.supervector_dimension // 2)

        ## N_ih*( m + D*z)
        # z is zero when the computation flow comes from update_X
        if latent_z is None:
            # Fn_x_ih = N_{i,h}*(o_{i,h} - m)
            fn_x_ih = f_i.flatten() - n_ic * (self.mean_supervector)
        else:
            # code goes here when the computation flow comes from compute_acculators
            # Fn_x_ih = N_{i,h}*(o_{i,h} - m - D*z_{i})
            fn_x_ih = f_i.flatten() - n_ic * (
                self.mean_supervector + self.D * latent_z[y_i]
            )

        """
        # JFA Part (eq 33)
        const blitz::Array<double, 1> &y = m_y[id];
        std::cout << "V" << y << std::endl;
        bob::math::prod(V, y, m_tmp_CD_b);
        m_cache_Fn_x_ih -= m_tmp_CD * m_tmp_CD_b;
        """

        return fn_x_ih

    def update_x(self, X, y, UProd, latent_x, latent_z=None):
        """
        Computes the accumulators U_a1, U_a2 for U
        U = A2 * A1^-1
        """

        # U.T @ inv(Sigma)
        UTinvSigma = self._computeUVD()
        # UProd = self.compute_uprod()

        session_offsets = np.zeros(self.estimate_number_of_classes(y))
        # For each sample
        for x_i, y_i in zip(X, y):
            id_plus_prod_ih = self._compute_id_plus_prod_ih(x_i, y_i, UProd)
            fn_x_ih = self._computefn_x_ih(x_i, y_i, latent_z)
            latent_x[y_i][:, int(session_offsets[y_i])] = id_plus_prod_ih @ (
                UTinvSigma @ fn_x_ih
            )
            session_offsets[y_i] += 1
        return latent_x

    def compute_uprod(self):
        """
        Computes U_c.T*inv(Sigma_c) @ U_c.T


        ### https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/da92d0e5799d018f311f1bf5cdd5a80e19e142ca/bob/learn/em/cpp/FABaseTrainer.cpp#L325
        """
        UProd = np.zeros((self.ubm.n_gaussians, self.r_U, self.r_U))
        for c in range(self.ubm.n_gaussians):
            # U_c.T
            U_c = self.U[
                c * self.feature_dimension : (c + 1) * self.feature_dimension, :
            ]
            sigma_c = self.ubm.variances[c].flatten()
            UProd[c, :, :] = U_c.T @ (U_c.T / sigma_c).T

        return UProd

    def compute_accumulators_U(self, X, y, UProd, latent_x, latent_z):
        ## U accumulators
        acc_U_A1 = np.zeros((self.ubm.n_gaussians, self.r_U, self.r_U))
        acc_U_A2 = np.zeros((self.supervector_dimension, self.r_U))

        # Loops over all people
        for y_i in set(y):
            # For each session
            for session_index, x_i in enumerate(
                self._get_statistics_by_class_id(X, y, y_i)
            ):
                id_plus_prod_ih = self._compute_id_plus_prod_ih(x_i, y_i, UProd)
                fn_x_ih = self._computefn_x_ih(x_i, y_i, latent_z)

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

    #################### Estimating D and z ######################

    def update_z(self, X, y, latent_x, latent_z, n_acc, f_acc):
        """
        Equation 38
        """

        # Precomputing
        # self.D.T / sigma
        dt_inv_sigma = self.D / self.variance_supervector
        # self.D.T / sigma * self.D
        dt_inv_sigma_d = dt_inv_sigma * self.D

        # for each class
        for y_i in set(y):

            id_plus_d_prod = self.computeIdPlusDProd_i(
                dt_inv_sigma_d, n_acc[y_i]
            )
            # X_i = X[y == y_i]  # Getting the statistics of the current class
            X_i = self._get_statistics_by_class_id(X, y, y_i)
            fn_z_i = self.compute_fn_z_i(
                X_i, y_i, latent_x, n_acc[y_i], f_acc[y_i]
            )
            latent_z[y_i] = self.updateZ_i(id_plus_d_prod, dt_inv_sigma, fn_z_i)

        return latent_z

    def updateZ_i(self, id_plus_d_prod, dt_inv_sigma, fn_z_i):
        """
        // Computes zi = Azi * D^T.Sigma^-1 * Fn_zi
        """
        return id_plus_d_prod * dt_inv_sigma * fn_z_i

    def computeIdPlusDProd_i(self, dt_inv_sigma_d, n_acc_i):
        """
        Computes: (I+Dt*diag(sigma)^-1*Ni*D)^-1

        Parameters
        ----------

         i: int
            Class id

        dt_inv_sigma_d: array
           Matrix representing `D.T / sigma`

        """

        tmp_CD = np.repeat(n_acc_i, self.supervector_dimension // 2)
        id_plus_d_prod = np.ones(tmp_CD.shape) + dt_inv_sigma_d * tmp_CD
        return 1 / id_plus_d_prod

    def compute_fn_z_i(self, X_i, y_i, latent_x, n_acc_i, f_acc_i):
        """
        Compute Fn_z_i = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h}) (Normalised first order statistics)

        Parameters
        ----------
            i: int
                Class id

        """

        U = self.U
        V = self.V  # Not doing the JFA

        latent_X_i = latent_x[y_i]
        m = self.mean_supervector

        # y = self.y[i] # Not doing JFA

        tmp_CD = np.repeat(n_acc_i, self.supervector_dimension // 2)

        ### NOT DOING JFA
        # bob::math::prod(V, y, m_tmp_CD_b);                 // m_tmp_CD_b = V * y
        # V_dot_v = V@v
        V_dot_v = 0  # Not doing JFA
        # m_cache_Fn_z_i = Fi - m_tmp_CD * (m + m_tmp_CD_b); // Fn_yi = sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i})
        fn_z_i = f_acc_i.flatten() - tmp_CD * (m - V_dot_v)

        # Looping over the sessions
        for session_id in range(len(X_i)):
            n_i = X_i[session_id].n
            tmp_CD = np.repeat(n_i, self.supervector_dimension // 2)
            x_i_h = latent_X_i[:, session_id]

            fn_z_i -= tmp_CD * (U @ x_i_h)

        return fn_z_i

    def initialize_XYZ(self, y):
        """
        Initialize E[x], E[y], E[z] state variables

        Eq. (38)
        latent_z = (n_classes, supervector_dimension)


        Eq. (37)
        latent_y =

        Eq. (36)
        latent_x = (n_classes, r_U, n_sessions)

        """

        # x (Eq. 36)
        # (n_classes, r_U,  n_samples )
        latent_x = []
        for y_i in set(y):
            latent_x.append(
                np.zeros(
                    (
                        self.r_U,
                        y.count(y_i),
                    )
                )
            )

        latent_y = None

        latent_z = np.zeros(
            (self.estimate_number_of_classes(y), self.supervector_dimension)
        )

        return latent_x, latent_y, latent_z

    # latent_x, latent_y, latent_z = self.initialize_XYZ(y)


class ISVMachine(FactorAnalysisBase):
    """
    Implements the Interssion Varibility Modelling hypothesis on top of GMMs

    Inter-Session Variability (ISV) modeling is a session variability modeling technique built on top of the Gaussian mixture modeling approach.
    It hypothesizes that within-class variations are embedded in a linear subspace in the GMM means subspace and these variations can be suppressed
    by an offset w.r.t each mean during the MAP adaptation.

    """

    def __init__(self, ubm, r_U, em_iterations, relevance_factor=4.0, seed=0):
        self.r_U = r_U
        self.seed = seed
        self.em_iterations = em_iterations
        super(ISVMachine, self).__init__(
            ubm, r_U=r_U, relevance_factor=relevance_factor
        )

    def initialize(self, X, y):
        return super(ISVMachine, self).initialize(X, y)

    def e_step(self, X, y, n_acc, f_acc):
        """
        E-step of the EM algorithm
        """
        # self.initialize_XYZ(y)
        UProd = self.compute_uprod()
        latent_x, latent_y, latent_z = self.initialize_XYZ(y)

        latent_x = self.update_x(X, y, UProd, latent_x)
        latent_z = self.update_z(X, y, latent_x, latent_z, n_acc, f_acc)
        acc_U_A1, acc_U_A2 = self.compute_accumulators_U(
            X, y, UProd, latent_x, latent_z
        )

        return acc_U_A1, acc_U_A2

    def m_step(self, acc_U_A1, acc_U_A2):
        """
        M-step of the EM algorithm
        """

        # Inverting A1 over the zero axis
        # https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        inv_A1 = np.linalg.inv(acc_U_A1)

        # Iterating over the gaussians to update U

        for c in range(self.ubm.n_gaussians):

            U_c = (
                acc_U_A2[
                    c
                    * self.feature_dimension : (c + 1)
                    * self.feature_dimension,
                    :,
                ]
                @ inv_A1[c, :, :]
            )
            self.U[
                c * self.feature_dimension : (c + 1) * self.feature_dimension,
                :,
            ] = U_c

        pass

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

        self.create_UVD(y)
        self.initialize(X, y)
        for i in range(self.em_iterations):
            logger.info("U Training: Iteration %d", i)
            acc_U_A1, acc_U_A2 = self.e_step(X, y)
            self.m_step(acc_U_A1, acc_U_A2)

        return self

    def enroll(self, X, iterations=1):
        """
        Enrolls a new client

        Parameters
        ----------
        X : numpy.ndarray
            Nxd features of N GMM statistics
        iterations : int
            Number of iterations to perform

        Returns
        -------
        self : object
            z

        """
        # We have only one class for enrollment
        y = list(np.zeros(len(X), dtype=np.int32))
        n_acc = self._sum_n_statistics(X, y=y)
        f_acc = self._sum_f_statistics(X, y=y)

        UProd = self.compute_uprod()
        latent_x, _, latent_z = self.initialize_XYZ(y)

        for i in range(iterations):
            logger.info("Enrollment: Iteration %d", i)
            # latent_x = self.update_x(X, y, UProd, [np.zeros((2, 2))])
            latent_x = self.update_x(X, y, UProd, latent_x, latent_z)
            latent_z = self.update_z(X, y, latent_x, latent_z, n_acc, f_acc)

        return latent_z


class JFAMachine(FactorAnalysisBase):
    """
    JFA
    """

    def __init__(self, ubm, r_U, em_iterations, relevance_factor=4.0, seed=0):
        self.r_U = r_U
        self.seed = seed
        self.em_iterations = em_iterations
        super(ISVMachine, self).__init__(
            ubm, r_U=r_U, relevance_factor=relevance_factor
        )

    def initialize(self, X, y):
        return super(ISVMachine, self).initialize(X, y)

    def e_step(self, X, y, n_acc, f_acc):
        """
        E-step of the EM algorithm
        """
        # self.initialize_XYZ(y)
        UProd = self.compute_uprod()
        latent_x, latent_y, latent_z = self.initialize_XYZ(y)

        latent_x = self.update_x(X, y, UProd, latent_x)
        latent_z = self.update_z(X, y, latent_x, latent_z, n_acc, f_acc)
        acc_U_A1, acc_U_A2 = self.compute_accumulators_U(
            X, y, UProd, latent_x, latent_z
        )

        return acc_U_A1, acc_U_A2

    def m_step(self, acc_U_A1, acc_U_A2):
        """
        M-step of the EM algorithm
        """

        # Inverting A1 over the zero axis
        # https://stackoverflow.com/questions/11972102/is-there-a-way-to-efficiently-invert-an-array-of-matrices-with-numpy
        inv_A1 = np.linalg.inv(acc_U_A1)

        # Iterating over the gaussinas to update U

        for c in range(self.ubm.n_gaussians):

            U_c = (
                acc_U_A2[
                    c
                    * self.feature_dimension : (c + 1)
                    * self.feature_dimension,
                    :,
                ]
                @ inv_A1[c, :, :]
            )
            self.U[
                c * self.feature_dimension : (c + 1) * self.feature_dimension,
                :,
            ] = U_c

        pass

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

        self.create_UVD(y)
        self.initialize(X, y)
        for i in range(self.em_iterations):
            logger.info("U Training: Iteration %d", i)
            acc_U_A1, acc_U_A2 = self.e_step(X, y)
            self.m_step(acc_U_A1, acc_U_A2)

        return self

    def enroll(self, X, iterations=1):
        """
        Enrolls a new client

        Parameters
        ----------
        X : numpy.ndarray
            Nxd features of N GMM statistics
        iterations : int
            Number of iterations to perform

        Returns
        -------
        self : object
            z

        """
        # We have only one class for enrollment
        y = list(np.zeros(len(X), dtype=np.int32))
        n_acc = self._sum_n_statistics(X, y=y)
        f_acc = self._sum_f_statistics(X, y=y)

        UProd = self.compute_uprod()
        latent_x, _, latent_z = self.initialize_XYZ(y)

        for i in range(iterations):
            logger.info("Enrollment: Iteration %d", i)
            # latent_x = self.update_x(X, y, UProd, [np.zeros((2, 2))])
            latent_x = self.update_x(X, y, UProd, latent_x, latent_z)
            latent_z = self.update_z(X, y, latent_x, latent_z, n_acc, f_acc)

        return latent_z
