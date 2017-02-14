# rubiks-cube-tracker
Analyze an image, directory of images, or video feed to locate a rubiks cube.
The RGB values for each square will be printed to STDOUT.

This works for 2x2x2, 3x3x3, 4x4x4, 5x5x5, and 6x6x6 cubes.  7x7x7 and larger
should also work if all of the square are the same size. I have not tested
larger than a 6x6x6.

If you are using the --webcam option a solution will be displayed on the screen
for 2x2x2, 3x3x3 and 4x4x4 cubes.  I have a solver for 5x5x5 but it takes about
a minute to compute the solution and the solutions tend to be 90+ steps so this
doesn't work very well for displaying on camera.


## Install
```
$ sudo apt-get install python-pip python3-pip
$ sudo pip install git+https://github.com/dwalton76/rubiks-cube-tracker.git
$ sudo pip install git+https://github.com/dwalton76/rubiks-color-resolver.git
$ sudo pip3 install git+https://github.com/dwalton76/rubiks-color-resolver.git
```

### Installing solvers
Solvers for various size cubes are available at https://github.com/dwalton76/rubiks-cube-solvers
Please follow the README instructions there to install the solvers you are interested in.


## How To Use

### Web Camera
- `--webcam 0` means use /dev/video0, `--webcam 1` means use /dev/video1, etc
- press the SPACEBAR to scan a side
- you MUST scan the sides in F R B L U D order
- press "r" to reset
```
$ rubiks-cube-tracker.py --webcam 0
```

### Single File
Analyze a single image.  This is only used for debugging and will pop up
images at various stages of locating the squares.
```
$ rubiks-cube-tracker.py --filename test/test-data/3x3x3-random-01/rubiks-side-B.png
```

### Directory of Files
Analyze a directory of images where the files are named rubiks-side-U.png, rubiks-side-L.png, etc
```
$ rubiks-cube-tracker.py --directory test/test-data/3x3x3-random-01/
```
