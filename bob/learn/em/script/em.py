"""The main entry for bob.em (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins
from bob.extension.scripts.click_helper import AliasedGroup


@with_plugins(pkg_resources.iter_entry_points("bob.em.cli"))
@click.group(cls=AliasedGroup)
def em():
    """Expected Maximization scripts."""
    pass
