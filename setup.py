from setuptools import setup, find_packages

setup(
    name='CAVE',
    version='0.1.0',
    packages=find_packages(include=['CAVE', 'CAVE.*']),
    install_requires=[
        "numpy",
        "torch",
    ]
)
