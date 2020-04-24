# import Libraries of other lib packages
import bob.io.base
import bob.math
import bob.learn.linear
import bob.sp

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.learn.em', __file__)

from ._library import *
from ._library import GMMMachine as _GMMMachine_C

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
  sameMatrix = numpy.ndarray((len(vect_a), len(vect_b)), 'bool')
  for j in range(len(vect_a)):
    for i in range(len(vect_b)):
      sameMatrix[j, i] = (vect_a[j] == vect_b[i])
  return sameMatrix


def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]


class GMMMachine(_GMMMachine_C):
    __doc__ = _GMMMachine_C.__doc__

    def __getstate__(self):        
        d = dict(self.__dict__)
        d["means"] = self.means
        d["variances"] = self.variances
        d["weights"] = self.weights
        return d

    def __setstate__(self, d):
        means = d["means"]
        self.__init__(means.shape[0], means.shape[1])
        self.means = means
        self.variances = d["variances"]
        self.means = d["means"]
        self.__dict__ = d
