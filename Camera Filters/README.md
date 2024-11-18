# Project Structure

## Directory Overview

- **bin:** This directory contains compiled binary executables generated from the source code. It includes the following executables:

  - `project1`: The main executable for the project.
  - `time_test`: An executable to run the timeBlur script.

- **scripts:** This directory holds shell scripts used to build and setup this project:

  - `build_project.sh`: This script is used to build the project, facilitating compilation and executable generation.
  - `setup_project.sh`: This script assists in setting up the project, handling any necessary configurations or dependencies.

## Usage

## Key Usage

- **'q':** Quit the application.
- **'s':** Save the current frame.
- **'g' to 'n':** Toggle between different video processing modes.
  - 'g': Grayscale
  - 'h': Custom Greyscale
  - 'p': Sepia
  - 'b': Blur 5x5
  - 'x': Sobel X 3x3
  - 'y': Sobel Y 3x3
  - 'm': Magnitude Sobel
  - 'i': Blur Quantize
  - 'f': Face Detection
  - 'c': Cartoonize
  - 'a': Face Blur
  - 'n': Negative Image
- **'v':** Toggle vignette effect.
- **'r':** Toggle video recording.


## Build
build_project.sh and setup_project.sh

Run the `setup_project.sh` and `build_project.sh` scripts in the scripts folder to build the project

```bash
./scripts/setup_project.sh
```

```bash
./scripts/build_project.sh
```
The project should be built and executables should be in the bin folder

### Executables (bin)

To run the main project executable for task1:

```bash
./bin/project1 task1 <path to any image file>
```

To run the main project executable for all other tasks:

```bash
./bin/project1 task2
```