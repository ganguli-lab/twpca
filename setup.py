from setuptools import setup, find_packages

setup(
    name='twpca',
    version='0.0.3',
    description='Time warped principal components analysis',
    author='Ben Poole, Alex Williams, Niru Maheswaranathan',
    url='https://github.com/ganguli-lab/twpca',
    install_requires=[
        'numpy',
        'tensorflow>=1.0.0',
        'scipy',
        'scikit-learn',
        'tqdm',
    ],
    extras_require={
        'dev': [],
        'test': ['flake8', 'pytest', 'coverage'],
    },
    long_description='''
        Simultaneous alignment and dimensionality reduction of mutlidimensional data
        ''',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering'],
    packages=find_packages(),
    license='MIT'
)
