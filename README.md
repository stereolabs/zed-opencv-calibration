# ZED OpenCV Calibration

A camera calibration toolkit for ZED cameras using OpenCV.

## Overview

This project provides two main applications for working with ZED cameras:

1. **Stereo Calibration Tool** - Interactive calibration data acquisition and processing
2. **Reprojection Viewer** - Real-time reprojection tool to visualize calibration results on unrectified images

## Requirements

### Dependencies

- **ZED SDK** (version 5.1 or higher)
- **OpenCV** (4.x recommended)
- **CUDA** (compatible with ZED SDK version)
- **OpenGL libraries**:
  - GLEW
  - FreeGLUT
  - OpenGL
- **CMake** (3.5 or higher)
- **C++17** compatible compiler

## Installation and building

```bash
git clone https://github.com/stereolabs/zed-opencv-calibration.git
cd zed-opencv-calibration

# Build stereo calibration tool and reprojection viewer
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Stereo Calibration

The calibration process requires a printed checkerboard pattern.

The default configuration expects a **[9x6 checkerboard with 24mm squared](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png/)**.

For other custom configurations, please use the command line options described below.

#### Prepare Calibration Target

- Print the checkerboard pattern and attach it on a rigid, flat surface.
- Ensure the pattern is perfectly flat and well-lit.
- Avoid reflections or glare on the checkerboard surface.

#### Run Calibration

Default command to start calibration:

```bash
cd build/stereo_calibration/
./zed_stereo_calibration
```

this command tries to open the first connected ZED camera for live calibration using the default checkerboard settings.

You can also specify different options to calibrate virtual stereo cameras or use custom checkerboard parameters:

```bash

Usage: ./zed_stereo_calibration [options]
  --h_edges <value>      Number of horizontal inner edges of the checkerboard
  --v_edges <value>      Number of vertical inner edges of the checkerboard
  --square_size <value>  Size of a square in the checkerboard (in mm)
  --svo <file>           Path to the SVO file.
  --fisheye              Use fisheye lens model.
  --virtual              Use ZED X One cameras as a virtual stereo pair.
  --left_id <id>         Id of the left camera if using virtual stereo.
  --right_id <id>        Id of the right camera if using virtual stereo.
  --left_sn <sn>         S/N of the left camera if using virtual stereo.
  --right_sn <sn>        S/N of the right camera if using virtual stereo.
  --help, -h             Show this help message.
```

#### Example Commands

- ZED Stereo Camera using an SVO file:

  `./zed_stereo_calibration --svo <full_path_to_svo_file>`

- Virtual Stereo Camera using camera IDs:

  `./zed_stereo_calibration --virtual --left_id 0 --right_id 1`

- Virtual Stereo Camera using camera serial numbers and a custom checkerboard (size 12x9 with 30mm squares):

  `./zed_stereo_calibration --virtual --left_sn <serial_number> --right_sn <serial_number> --h_edges 12 --v_edges 9 --square_size 30.0`

- Virtual Stereo Camera with **fisheye lenses** using camera serial numbers:

  `./zed_stereo_calibration --fisheye --virtual --left_sn <serial_number> --right_sn <serial_number>`

>:pushpin: **Note**: You can easily obtain the serial numbers or the IDs of your connected ZED cameras by running the following command:
>
> ```bash
> ZED_Explorer --all
> ```

#### Calibration Process

The calibration process consists of two main phases:

1. **Data Acquisition**: Move the checkerboard in front of the camera(s) to capture diverse views. The tool provides real-time feedback on the quality of the captured data.
2. **Calibration Computation**: Once sufficient data is collected, the tool computes the calibration parameters and saves them to two files.

The **Data Acquisition** phase consists of moving the checkerboard in front of the camera(s) to capture diverse views. The tool provides real-time feedback on the quality of the captured data regarding *XY coverage*, *distance variation*, and *skewness*.

When the checkerboard is placed in a position that you want to capture, press the **Spacebar** or the **S** key to capture the images.

- If the checkerboard is detected in both images, and the captured data are different enough from the previously captured images, the data is accepted, and the quality indicators are updated.
- If the data is not accepted, a message is displayed in the GUI output indicating the reason (e.g., checkerboard not detected, not enough variation, etc.).

In order to collect good calibration data, ensure that:

- The checkerboard is always fully visible in both left and right images. Corners detected in both images are highlighted with colored visual markers.
- The checkerboard moves over a wide area of the image frame. "Green" polygons appear on the left image to indicate the covered areas. When one of the 4 zones of the left image becomes fully green, the coverage requirement is met for that part of the image.
- The checkerboard is moved closer and farther from the camera to ensure depth variation. At least one image covering almost the full left frame is required.
- The checkerboard is tilted and rotated to provide different angles.

The "Horizontal Coverage", "Vertical Coverage", "Checkerboard sizes", and "Checkerboard skews" percentages indicates the quality of the collected data for each criterion. When all criteria reach 100%, a minimum number of images is collected, or a maximum number of images is reached, the "Calibrate" process will automatically start.

You can follow the steps of the calibration process in the terminal output:

1. The left camera is calibrated first, followed by the right camera to obtain the intrinsic parameters.
2. Finally, the stereo calibration is performed to compute the extrinsic parameters between the two cameras.

Good calibration results typically yield a reprojection error below 0.5 pixels for each calibration step.

In case one of the reprojection errors is too high, the result of the calibration is not accurate enough, and you should redo the calibration process verifying that the checkerboard is flat and well-lit, the lenses of the cameras are clean, and that the light of the environment is stable and not generating reflections or glares on the checkerboard.

After a good calibration is complete, two files are generated:

- `zed_calibration_<serial_number>.yml`: Contains intrinsic and extrinsic parameters for the stereo camera setup ins OpenCV format.
- `SN<serial_number>.conf`: Contains the calibration parameters in ZED SDK format.

You can use these files in your ZED SDK applications:

- [Use the `sl::InitParameters::optional_opencv_calibration_file` parameter to load the calibration from the OpenCV file](https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html#a9eab2753374ef3baec1d31960859ba19).
- Manually copy the `SN<serial_number>.conf` file to the ZED SDK calibration folder to make the ZED SDK automatically use it:

  - Linux: `/usr/local/zed/settings/`
  - Windows: `C:\ProgramData\Stereolabs\settings`
- [Use the `sl::InitParameters::optional_settings_path` to indicate to the ZED SDK where to find the custom `SN<serial_number>.conf` calibration file](https://www.stereolabs.com/docs/api/structsl_1_1InitParameters.html#aa8262e36d2d4872410f982a735b92294).

>:pushpin: **Note**: When calibrating a virtual ZED X One stereo rig, the serial number of the Virtual Stereo Camera is generated by the ZED SDK using the serial numbers of the two individual cameras. Make sure to use this generated serial number when loading the calibration in your application to have a unique identifier for the virtual stereo setup.

### Reprojection Viewer

Visualize the rectification of the image on the full

```bash
cd reprojection_viewer/build
./ZED_Depth_Repro [options]
```
