# import Libraries of other lib packages
import bob.io.base
import bob.math
import bob.learn.linear
import bob.sp

# import our own Library
import bob.extension

bob.extension.load_bob_library("bob.learn.em", __file__)

from ._library import *
from ._library import GMMMachine as _GMMMachine_C
from ._library import ISVBase as _ISVBase_C
from ._library import ISVMachine as _ISVMachine_C
from ._library import KMeansMachine as _KMeansMachine_C
from ._library import GMMStats as _GMMStats_C
from ._library import IVectorMachine as _IVectorMachine_C

from . import version
from .version import module as __version__
from .version import api as __api_version__
from .train import *


def get_config():
    """Returns a string containing the configuration information.
    """
    return bob.extension.get_config(__name__, version.externals, version.api)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]


class GMMMachine(_GMMMachine_C):
    __doc__ = _GMMMachine_C.__doc__

    def update_dict(self, d):
        self.means = d["means"]
        self.variances = d["variances"]
        self.weights = d["weights"]

    @staticmethod
    def gmm_shape_from_dict(d):
        return d["means"].shape

    @classmethod
    def create_from_dict(cls, d):
        shape = GMMMachine.gmm_shape_from_dict(d)
        gmm_machine = cls(shape[0], shape[1])
        gmm_machine.update_dict(d)
        return gmm_machine

    @staticmethod
    def to_dict(gmm_machine):
        gmm_data = dict()
        gmm_data["means"] = gmm_machine.means
        gmm_data["variances"] = gmm_machine.variances
        gmm_data["weights"] = gmm_machine.weights
        return gmm_data

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        shape = self.gmm_shape_from_dict(d)
        self.__init__(shape[0], shape[1])
        self.update_dict(d)


class ISVBase(_ISVBase_C):
    __doc__ = _ISVBase_C.__doc__

    @staticmethod
    def to_dict(isv_base):
        isv_data = dict()
        isv_data["gmm"] = GMMMachine.to_dict(isv_base.ubm)
        isv_data["u"] = isv_base.u
        isv_data["d"] = isv_base.d

        return isv_data

    def update_dict(self, d):
        ubm = GMMMachine.create_from_dict(d["gmm"])
        u = d["u"]
        self.__init__(ubm, u.shape[1])
        self.u = u
        self.d = d["d"]

    @classmethod
    def create_from_dict(cls, d):
        ubm = GMMMachine.create_from_dict(d["gmm"])
        ru = d["u"].shape[1]
        isv_base = ISVBase(ubm, ru)
        isv_base.u = d["u"]
        isv_base.d = d["d"]
        return isv_base

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.update_dict(d)


class ISVMachine(_ISVMachine_C):
    __doc__ = _ISVMachine_C.__doc__

    @staticmethod
    def to_dict(isv_machine):
        isv_data = dict()
        isv_data["x"] = isv_machine.x
        isv_data["z"] = isv_machine.z
        isv_data["isv_base"] = ISVBase.to_dict(isv_machine.isv_base)

        return isv_data

    def update_dict(self, d):
        isv_base = ISVBase.create_from_dict(d["isv_base"])
        self.__init__(isv_base)
        self.x = d["x"]
        self.z = d["z"]

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.update_dict(d)


class KMeansMachine(_KMeansMachine_C):
    __doc__ = _KMeansMachine_C.__doc__

    @staticmethod
    def to_dict(kmeans_machine):
        kmeans_data = dict()
        kmeans_data["means"] = kmeans_machine.means
        return kmeans_data

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        means = d["means"]
        self.__init__(means.shape[0], means.shape[1])
        self.means = means


class GMMStats(_GMMStats_C):
    __doc__ = _GMMStats_C.__doc__

    @staticmethod
    def to_dict(gmm_stats):
        gmm_stats_data = dict()
        gmm_stats_data["log_likelihood"] = gmm_stats.log_likelihood
        gmm_stats_data["t"] = gmm_stats.t
        gmm_stats_data["n"] = gmm_stats.n
        gmm_stats_data["sum_px"] = gmm_stats.sum_px
        gmm_stats_data["sum_pxx"] = gmm_stats.sum_pxx
        return gmm_stats_data

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        shape = d["sum_pxx"].shape
        self.__init__(shape[0], shape[1])
        self.t = d["t"]
        self.n = d["n"]
        self.log_likelihood = d["log_likelihood"]
        self.sum_px = d["sum_px"]
        self.sum_pxx = d["sum_pxx"]


class IVectorMachine(_IVectorMachine_C):
    __doc__ = _IVectorMachine_C.__doc__

    @staticmethod
    def to_dict(ivector_machine):
        ivector_data = dict()
        ivector_data["gmm"] = GMMMachine.to_dict(ivector_machine.ubm)
        ivector_data["sigma"] = ivector_machine.sigma
        ivector_data["t"] = ivector_machine.t

        return ivector_data

    def update_dict(self, d):
        ubm = GMMMachine.create_from_dict(d["gmm"])
        t = d["t"]
        self.__init__(ubm, t.shape[1])
        self.sigma = d["sigma"]
        self.t = t

    def __getstate__(self):
        d = dict(self.__dict__)
        d.update(self.__class__.to_dict(self))
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.update_dict(d)
