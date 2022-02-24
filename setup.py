#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()


setup(
    name="bob.learn.em",
    version=open("version.txt").read().rstrip(),
    description="Bindings for EM machines and trainers of Bob",
    url="http://gitlab.idiap.ch/bob/bob.learn.em",
    license="BSD",
    author="Andre Anjos",
    author_email="andre.anjos@idiap.ch",
    keywords="bob, em, expectation-maximization",
    long_description=open("README.rst").read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requires,
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
