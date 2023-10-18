from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements
setup(
    name='QIP',
    version='0.1',
    description='Quantum Image Processing Package',
    long_description='This package contains tools to represent and manipulate images in Qiskit.',
    author='Giacomo Antonioli',
    author_email='giaco.antonioli@gmail.com',
    packages=find_packages(),
    install_reqs= parse_requirements('requirements.txt', session='hack'),
)