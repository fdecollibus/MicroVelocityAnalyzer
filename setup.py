from setuptools import setup, find_packages

setup(
    name='micro_velocity_analyzer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pyspark',
        
    ],
    entry_points={
        'console_scripts': [
            'micro-velocity-analyzer=micro_velocity_analyzer.micro_velocity_analyzer:main',
        ],
    },
    description='A package for analyzing MicroVelocity based on transfer data.',
    author='Francesco Maria De Collibus',
    author_email='francesco.decollibus@business.uzh.ch',
    url='https://github.com/fdecollibus/MicroVelocityAnalyzer',
)

