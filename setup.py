from setuptools import setup

setup(
    name='rubikscubetracker',
    version='1.0.0',
    description='Extract rubiks cube RGB values from an image/video',
    keywords='rubiks cube color opencv',
    url='https://github.com/dwalton76/rubiks-cube-tracker',
    author='dwalton76',
    author_email='dwalton76@gmail.com',
    license='GPLv3',
    scripts=['usr/bin/rubiks-cube-tracker.py'],
    packages=['rubikscubetracker'],
)
