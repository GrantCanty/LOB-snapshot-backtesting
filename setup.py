from setuptools import setup

setup(
    name='LOB-snapshot-backtesting',
    version='0.0.1',
    description='gymnasium environment for testing 1 second snap shots of limit order books',
    url='git@github.com:GrantCanty/LOB-snapshot-backtesting.git',
    author='Grant Canty',
    author_email='gc933815@ohio.edu',
    licence='unlicense',
    packages=['environment'],
    zip_safe=False
)