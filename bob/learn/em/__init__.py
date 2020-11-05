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

from . import version
from .version import module as __version__
from .version import api as __api_version__
from .train import *


def ztnorm_same_value(vect_a, vect_b):
    """Computes the matrix of boolean D for the ZT-norm, which indicates where
     the client ids of the T-Norm models and Z-Norm samples match.

     vect_a An (ordered) list of client_id corresponding to the T-Norm models
     vect_b An (ordered) list of client_id corresponding to the Z-Norm impostor samples
     """
    import numpy

    sameMatrix = numpy.ndarray((len(vect_a), len(vect_b)), "bool")
    for j in range(len(vect_a)):
        for i in range(len(vect_b)):
            sameMatrix[j, i] = vect_a[j] == vect_b[i]
    return sameMatrix


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
        self.means = d["means"]

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
