#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core', 'bob.io.base', 'bob.sp', 'bob.math', 'bob.learn.activation', 'bob.learn.linear']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['boost']
boost_modules = ['system', 'python']

setup(

    name='bob.learn.misc',
    version=version,
    description='Bindings for miscelaneous machines and trainers of Bob',
    url='http://github.com/bioidiap/bob.learn.misc',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    setup_requires = build_requires,
    install_requires = build_requires,

    namespace_packages=[
      "bob",
      "bob.learn",
    ],

    ext_modules = [
      Extension("bob.learn.misc.version",
        [
          "bob/learn/misc/version.cpp",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),

      Library("bob.learn.misc.bob_learn_misc",
        [
          "bob/learn/misc/cpp/BICMachine.cpp",
          "bob/learn/misc/cpp/Gaussian.cpp",
          "bob/learn/misc/cpp/GMMMachine.cpp",
          "bob/learn/misc/cpp/GMMStats.cpp",
          "bob/learn/misc/cpp/IVectorMachine.cpp",
          "bob/learn/misc/cpp/JFAMachine.cpp",
          "bob/learn/misc/cpp/KMeansMachine.cpp",
          "bob/learn/misc/cpp/LinearScoring.cpp",
          "bob/learn/misc/cpp/PLDAMachine.cpp",
          "bob/learn/misc/cpp/WienerMachine.cpp",
          "bob/learn/misc/cpp/ZTNorm.cpp",

          "bob/learn/misc/cpp/BICTrainer.cpp",
          "bob/learn/misc/cpp/EMPCATrainer.cpp",
          "bob/learn/misc/cpp/GMMTrainer.cpp",
          "bob/learn/misc/cpp/IVectorTrainer.cpp",
          "bob/learn/misc/cpp/JFATrainer.cpp",
          "bob/learn/misc/cpp/KMeansTrainer.cpp",
          "bob/learn/misc/cpp/MAP_GMMTrainer.cpp",
          "bob/learn/misc/cpp/ML_GMMTrainer.cpp",
          "bob/learn/misc/cpp/PLDATrainer.cpp",
          "bob/learn/misc/cpp/WienerTrainer.cpp",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),

#      Extension("bob.learn.misc._library",
#        [
#          "bob/learn/misc/old/bic.cc",
#
#          # external requirements as boost::python bindings
#          "bob/learn/misc/old/blitz_numpy.cc",
#          "bob/learn/misc/old/ndarray.cc",
#          "bob/learn/misc/old/ndarray_numpy.cc",
#          "bob/learn/misc/old/tinyvector.cc",
#          "bob/learn/misc/old/random.cc",
#
#          "bob/learn/misc/old/main.cc",
#        ],
#        bob_packages = bob_packages,
#        packages = packages,
#        boost_modules = boost_modules,
#        version = version,
#      ),

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

          "bob/learn/misc/old/main.cc",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),
    ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
