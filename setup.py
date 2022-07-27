from setuptools import find_packages, setup

setup(
    name='vaccine_clinic',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version='0.1.0',
    description='SimPy model of vaccine clinic',
    author='misken',
    license='MIT',
)
