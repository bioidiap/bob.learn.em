#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.io.base']))
from bob.blitz.extension import Extension
import bob.io.base

import os
include_dirs = [bob.io.base.get_include()]

packages = ['bob-machine >= 2.0.0a2', 'bob-trainer >= 2.0.0a2', 'boost']
version = '2.0.0a0'

setup(

    name='bob.learn.misc',
    version=version,
    description='Bindings for miscelaneous machines and trainers',
    url='http://github.com/bioidiap/bob.learn.misc',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.core',
      'bob.io.base',
      'bob.sp',
      ],

    namespace_packages=[
      "bob",
      "bob.learn",
      ],

    ext_modules = [
      Extension("bob.learn.misc.version",
        [
          "bob/learn/misc/version.cpp",
          ],
        packages = packages,
        include_dirs = include_dirs,
        version = version,
        ),
      Extension("bob.learn.misc._old_library",
        [
          "bob/learn/misc/old/bic.cc",
          "bob/learn/misc/old/bic_trainer.cc",
          "bob/learn/misc/old/empca_trainer.cc",
          "bob/learn/misc/old/gaussian.cc",
          "bob/learn/misc/old/gmm.cc",
          "bob/learn/misc/old/gmm_trainer.cc",
          "bob/learn/misc/old/ivector.cc",
          "bob/learn/misc/old/ivector_trainer.cc",
          "bob/learn/misc/old/jfa.cc",
          "bob/learn/misc/old/jfa_trainer.cc",
          "bob/learn/misc/old/kmeans.cc",
          "bob/learn/misc/old/kmeans_trainer.cc",
          "bob/learn/misc/old/machine.cc",
          "bob/learn/misc/old/linearscoring.cc",
          "bob/learn/misc/old/plda.cc",
          "bob/learn/misc/old/plda_trainer.cc",
          "bob/learn/misc/old/wiener.cc",
          "bob/learn/misc/old/wiener_trainer.cc",
          "bob/learn/misc/old/ztnorm.cc",

          # external requirements as boost::python bindings
          "bob/learn/misc/old/blitz_numpy.cc",
          "bob/learn/misc/old/ndarray.cc",
          "bob/learn/misc/old/ndarray_numpy.cc",
          "bob/learn/misc/old/tinyvector.cc",
          "bob/learn/misc/old/hdf5.cc",
          "bob/learn/misc/old/random.cc",

          "bob/learn/misc/old/main.cc",
        ],
        packages = packages,
        boost_modules = ['python'],
        include_dirs = include_dirs,
        version = version,
        ),
      ],

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
