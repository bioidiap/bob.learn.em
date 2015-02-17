# import Libraries of other lib packages
import bob.io.base
import bob.math
import bob.learn.linear

# import our own Library
import bob.extension
bob.extension.load_bob_library('bob.learn.em', __file__)

#from ._old_library import *
from ._library import *
from . import version
from .version import module as __version__
from .__MAP_gmm_trainer__ import *
from train import *

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

  import pkg_resources
  from .version import externals

  packages = pkg_resources.require(__name__)
  this = packages[0]
  deps = packages[1:]

  retval =  "%s: %s (%s)\n" % (this.key, this.version, this.location)
  retval += "  - c/c++ dependencies:\n"
  for k in sorted(externals): retval += "    - %s: %s\n" % (k, externals[k])
  retval += "  - python dependencies:\n"
  for d in deps: retval += "    - %s: %s (%s)\n" % (d.key, d.version, d.location)

  return retval.strip()

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
