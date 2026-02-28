from setuptools import setup, find_packages

setup(
    name='LOB-snapshot-backtesting',
    version='0.0.1',
    description='gymnasium environment for testing 1 second snap shots of limit order books',
    url='https://github.com/GrantCanty/LOB-snapshot-backtesting.git',
    author='Grant Canty',
    author_email='gc933815@ohio.edu',
    license='unlicense',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'stable-baselines3',
        'numpy',
        'torch',
        'pytz'
    ]
)