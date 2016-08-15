#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

bob_packages = ['bob.core', 'bob.io.base', 'bob.sp', 'bob.math', 'bob.learn.activation', 'bob.learn.linear']

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.extension>=2.0.7', 'bob.blitz'] + bob_packages))
from bob.blitz.extension import Extension, Library, build_ext

from bob.extension.utils import load_requirements
build_requires = load_requirements()

# Define package version
version = open("version.txt").read().rstrip()

packages = ['boost']
boost_modules = ['system']

setup(

    name='bob.learn.em',
    version=version,
    description='Bindings for EM machines and trainers of Bob',
    url='http://gitlab.idiap.ch/bob/bob.learn.em',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    setup_requires = build_requires,
    install_requires = build_requires,



    ext_modules = [
      Extension("bob.learn.em.version",
        [
          "bob/learn/em/version.cpp",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),

      Library("bob.learn.em.bob_learn_em",
        [
          "bob/learn/em/cpp/Gaussian.cpp",
          "bob/learn/em/cpp/GMMMachine.cpp",
          "bob/learn/em/cpp/GMMStats.cpp",
          "bob/learn/em/cpp/IVectorMachine.cpp",
          "bob/learn/em/cpp/KMeansMachine.cpp",
          "bob/learn/em/cpp/LinearScoring.cpp",
          "bob/learn/em/cpp/PLDAMachine.cpp",
          "bob/learn/em/cpp/ZTNorm.cpp",

          "bob/learn/em/cpp/FABase.cpp",
          "bob/learn/em/cpp/JFABase.cpp",
          "bob/learn/em/cpp/ISVBase.cpp",
          "bob/learn/em/cpp/JFAMachine.cpp",
          "bob/learn/em/cpp/ISVMachine.cpp",

          "bob/learn/em/cpp/FABaseTrainer.cpp",
          "bob/learn/em/cpp/JFATrainer.cpp",
          "bob/learn/em/cpp/ISVTrainer.cpp",

          "bob/learn/em/cpp/EMPCATrainer.cpp",
          "bob/learn/em/cpp/GMMBaseTrainer.cpp",
          "bob/learn/em/cpp/IVectorTrainer.cpp",
          "bob/learn/em/cpp/KMeansTrainer.cpp",
          "bob/learn/em/cpp/MAP_GMMTrainer.cpp",
          "bob/learn/em/cpp/ML_GMMTrainer.cpp",
          "bob/learn/em/cpp/PLDATrainer.cpp",
        ],
        bob_packages = bob_packages,
        packages = packages,
        boost_modules = boost_modules,
        version = version,
      ),

      Extension("bob.learn.em._library",
        [
          "bob/learn/em/gaussian.cpp",
          "bob/learn/em/gmm_stats.cpp",
          "bob/learn/em/gmm_machine.cpp",
          "bob/learn/em/kmeans_machine.cpp",
          "bob/learn/em/kmeans_trainer.cpp",

          "bob/learn/em/ml_gmm_trainer.cpp",
          "bob/learn/em/map_gmm_trainer.cpp",

          "bob/learn/em/jfa_base.cpp",
          "bob/learn/em/jfa_machine.cpp",
          "bob/learn/em/jfa_trainer.cpp",

          "bob/learn/em/isv_base.cpp",
          "bob/learn/em/isv_machine.cpp",
          "bob/learn/em/isv_trainer.cpp",

          "bob/learn/em/ivector_machine.cpp",
          "bob/learn/em/ivector_trainer.cpp",

          "bob/learn/em/plda_base.cpp",
          "bob/learn/em/plda_machine.cpp",

          "bob/learn/em/empca_trainer.cpp",

          "bob/learn/em/plda_trainer.cpp",

          "bob/learn/em/ztnorm.cpp",

          "bob/learn/em/linear_scoring.cpp",

          "bob/learn/em/main.cpp",
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
