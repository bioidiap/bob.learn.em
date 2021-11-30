import bob.extension


def get_config():
    """Returns a string containing the configuration information."""
    return bob.extension.get_config(__name__)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith("_")]
