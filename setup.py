from setuptools import setup, find_packages

setup(
    name='dspyplot',
    version='0.1',
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'librosa',
        'pathlib',
        'SoundFile',
        'pyloudnorm'
    ],
    packages=find_packages(),
    description='A library for plotting digital signal processing data.'
                ' Removes a lot of boilerplate from DSP code. '
                'Focuses mostly on audio signals.',
    author='Jan Wilczek',
    author_email='jan.wilczek@thewolfsound.com',
    url='https://github.com/JanWilczek/dspyplot',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)
