# rubiks-cube-tracker
Analyze an image, directory of images, or video feed to locate a rubiks cube.
The RGB values for each square will be printed to STDOUT.

This works for 2x2x2, 3x3x3, 4x4x4, 5x5x5, and 6x6x6 cubes.  6x6x6 is the
largest cube I have test with but 7x7x7 and larger should also work if all
of the squares are the same size.

If you are using the --webcam option a solution will be displayed on the screen
for 2x2x2, 3x3x3 and 4x4x4 cubes.  I have a solver for 5x5x5 but it takes about
a minute to compute the solution and the solutions tend to be 90+ steps so this
doesn't work very well for displaying on camera.


## Install
```
$ sudo apt-get install python-pip python-opencv
$ sudo pip install git+https://github.com/dwalton76/rubiks-cube-tracker.git
```

### Installing rubiks-color-resolver
Follow the instructions at https://github.com/dwalton76/rubiks-color-resolver
to install the rubiks-color-resolver library

### Installing solvers
Follow the instructions at https://github.com/dwalton76/rubiks-cube-NxNxN-solver
to install a solver for 2x2x2, 3x3x3, 4x4x4, 5x5x5, 6x6x6, and 7x7x7 cubes


## How To Use

### Web Camera

```
$ rubiks-cube-tracker.py --webcam 0
```

- `--webcam 0` means use /dev/video0, `--webcam 1` means use /dev/video1, etc
- Press the SPACEBAR to scan a side
- You MUST scan the sides in F R B L U D order. Once you've scanned L you want to put F back in front and then flip forward one time to scan U, this way U is oriented correctly.  To go from U to D just flip forward two times. Once you've scanned D, D will be facing the camera, U will be facing you and F will be on top.
- Flip the cube so that U is on top and F is facing you, then follow the solve steps on the screen.
- Press "r" to reset

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
