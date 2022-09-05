# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='STEMPLATE',
    version='0.1.0',
    description='project description here',
    long_description=readme,
    author='Lachlan Casey',
    author_email='lachlanc.612@gmail.com',
    url='https://github.com/lachlanc61/STEMPLATE',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

