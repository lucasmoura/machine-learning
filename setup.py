from setuptools import setup, find_packages

setup(
    version='0.1',
    description='Implementation of some machine learning algorithms',
    author='Lucas Moura',
    author_email='nate@natereed.com',
    packages=find_packages(),
    install_requires=['numpy'],
    test_suite="tests",
    )
