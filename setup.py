#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, Extension, find_packages
from setuptools.dist import Distribution


with open('README.rst') as file:
    long_description = file.read()

setup(name='impvol',
      version='1.0',
      description=('Compute Black-Scholes implied volatility'),
      long_description=long_description,
      author='Stanislav Khrapov',
      license='NCSA',
      author_email='khrapovs@gmail.com',
      url='https://github.com/khrapovs/impvol',
      py_modules=['impvol'],
      package_dir={'impvol': './impvol'},
      packages=find_packages(),
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
      ],
)
