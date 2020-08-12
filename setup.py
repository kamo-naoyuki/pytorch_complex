#!/usr/bin/env python
import os
from setuptools import find_packages
from setuptools import setup

dirname = os.path.dirname(__file__)
setup(name='torch_complex',
      version='0.1.2',
      description='A fugacious python class for PyTorch-ComplexTensor',
      long_description=open(os.path.join(dirname, 'README.md'),
                            encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      author='Naoyuki Kamo',
      author_email='naoyuki.kamo829@gmail.com',
      url='https://github.com/kamo-naoyuki/torch_complex',
      packages=find_packages(include=['torch_complex']),
      install_requires=['numpy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'pytest-cov'],
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'Operating System :: POSIX :: Linux',
          'License :: OSI Approved :: Apache Software License',
          'Topic :: Software Development :: Libraries :: Python Modules'],
      )
