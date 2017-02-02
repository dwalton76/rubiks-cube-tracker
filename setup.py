from setuptools import setup


setup(
    name='rubikssquareextractor',
    version='1.0.0',
    description='Extract rubiks cube RGB values from an image with a cube',
    keywords='rubiks cube color',
    url='https://github.com/dwalton76/rubiks-square-extractor',
    author='dwalton76',
    author_email='dwalton76@gmail.com',
    license='GPLv3',
    scripts=['usr/bin/rubiks-square-extractor.py'],
    packages=['rubikssquareextractor'],
)
