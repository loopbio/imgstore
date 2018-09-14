#!/usr/bin/env python
# coding=utf-8

# Authors: John Stowers <john@loopbio.com>, Santi Villalba <santi@loopbio.com>
# Licence: BSD 3 clause

from setuptools import setup, find_packages

# Make pypi rst-happy
try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError):
    pypandoc = None
    long_description = open('README.md').read()

setup(
    name='imgstore',
    license='BSD 3 clause',
    description='IMGStore houses your video frames',
    long_description=long_description,
    version='0.2.0',
    url='https://github.com/loopbio/imgstore',
    author='John Stowers, Santi Villalba',
    author_email='john@loopbio.com, santi@loopbio.com',
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'numpy',
        'pandas',
        'pyyaml',
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-pep8'
    ],
    extras_require={
        'bloscpack': ['bloscpack']
    },
    entry_points={
        'console_scripts': [
            'imgstore-view = imgstore.util:main_viewer',
            'imgstore-save = imgstore.util:main_saver',
            'imgstore-test = imgstore.util:main_test',
        ]
    },
)
