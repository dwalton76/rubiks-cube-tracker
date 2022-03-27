# third party libraries
from setuptools import setup

setup(
    name="rubikscubetracker",
    version="3.0.0",
    description="Extract rubiks cube RGB values from an image/video",
    keywords="rubiks cube color opencv",
    url="https://github.com/dwalton76/rubiks-cube-tracker",
    author="Daniel Walton",
    author_email="dwalton76@gmail.com",
    license_files=("LICENSE",),
    scripts=["usr/bin/rubiks-cube-tracker.py"],
    packages=["rubikscubetracker"],
    install_requires=["opencv-python>=4.5"],
)
