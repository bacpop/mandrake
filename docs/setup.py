#!/usr/bin/env python
# vim: set fileencoding=<utf-8> :

from codecs import open
from os import path
import os
import re
import io

from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


here = path.abspath(path.dirname(__file__))


with open(path.join(here, '../README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mandrake',
    version=find_version("../mandrake/__init__.py"),
    description='Visualisation of pathogen population structure',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/johnlees/mandrake',
    author='John Lees',
    author_email='john@johnlees.me',
    license='Apache Software License',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.7.0',
    keywords='bacteria genomics population-genetics k-mer visualisation',
    packages=['mandrake'],
    entry_points={
        "console_scripts": [
            'mandrake = mandrake.__main__:main'
            ]
    },
    test_suite="test",
)
