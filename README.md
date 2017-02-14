# rubiks-cube-tracker
Given an image or video feed with a rubiks cube
- find the RGB value for each square
- display a solution (2x2x2, 3x3x3, and 4x4x4 only)

## Install

### Installing rubiks-cube-tracker
```
$ sudo apt-get install python-pip
$ sudo pip install git+https://github.com/dwalton76/rubiks-cube-tracker.git

```

### Installing rubiks-color-resolver
When the cube is scanned we get the RGB (red, green, blue) value for 
all squares of the cube.  rubiks-color-resolver analyzes those RGB 
values to determine which of the six possible cube colors is the color for 
each square.
```
$ sudo apt-get install python3-pip
$ sudo pip3 install git+https://github.com/dwalton76/rubiks-color-resolver.git

