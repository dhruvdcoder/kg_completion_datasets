from setuptools import setup, find_packages
from pathlib import Path

requirements_file = Path('requirements.txt')

if (not requirements_file.exists()) or (not requirements_file.is_file()):
    raise Exception("No requirements.txt found")
with open(requirements_file) as f:
    install_requires = list(f.read().splitlines())

setup(
    name='datasets',
    version='0.0.1',
    description='AllenNLP style data pipeline for KB Completion',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'datasets': ['py.typed']},
    install_requires=install_requires,
    zip_safe=False)
