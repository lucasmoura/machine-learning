from setuptools import setup, find_packages

setup(
    name='machine_learning',
    version='0.1',
    description='Implementation of some machine learning algorithms',
    author='Lucas Moura',
    author_email='lucas.moura128@gmail.com',
    packages=find_packages(),
    install_requires=['numpy'],
    test_suite="tests",
    )
