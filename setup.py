#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup


setup(name='torch_complex',
      version='0.0.1',
      description='A fugacious python class for PyTorch-ComplexTensor',
      author='Naoyuki Kamo',
      author_email='naoyuki.kamo829@gmail.com',
      url='https://github.com/kamo-naoyuki/torch_complex',
      packages=find_packages(include=['torch_complex']),
      install_requires=['numpy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov']
      )
