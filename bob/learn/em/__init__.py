import bob.extension

from .factor_analysis import ISVMachine, JFAMachine
from .gmm import GMMMachine, GMMStats
from .kmeans import KMeansMachine
from .linear_scoring import linear_scoring  # noqa: F401
from .wccn import WCCN
from .whitening import Whitening


def get_config():
    """Returns a string containing the configuration information."""
    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
    """Says object was actually declared here, an not on the import module.

    Parameters:

      *args: An iterable of objects to modify

    Resolves `Sphinx referencing issues
    <https://github.com/sphinx-doc/sphinx/issues/3048>`
    """

    for obj in args:
        obj.__module__ = __name__


__appropriate__(
    KMeansMachine, GMMMachine, GMMStats, WCCN, Whitening, ISVMachine, JFAMachine
)
__all__ = [_ for _ in dir() if not _.startswith("_")]
