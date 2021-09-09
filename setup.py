# File: setup.py
from setuptools import find_packages, setup

setup(
    use_scm_version=True,
    name="globalHelioseismology",
    packages=find_packages(where="src"),
    package_dir={"", "src"},
)
