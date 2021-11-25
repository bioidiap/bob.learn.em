#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import dist
from setuptools import setup

from bob.extension.utils import find_packages
from bob.extension.utils import load_requirements

bob_packages = ['bob.core', 'bob.io.base', 'bob.sp', 'bob.math', 'bob.learn.activation', 'bob.learn.linear']
dist.Distribution(dict(setup_requires=['bob.extension>=2.0.7'] + bob_packages))


install_requires = load_requirements()

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
    keywords="bob, em, expectation-maximization",

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires = install_requires,

    classifiers=[
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
