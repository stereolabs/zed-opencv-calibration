# ZED OpenCV Calibration

A stereo camera calibration toolkit for ZED cameras using OpenCV.

## Overview

This project provides two main applications for working with ZED cameras:

1. **Stereo Calibration Tool** - Interactive calibration data acquisition and processing
2. **Reprojection Viewer** - Real-time reprojection tool to visualize calibration on unrectified images

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

## Installation

```bash
git clone https://github.com/stereolabs/zed-opencv-calibration.git
cd zed-opencv-calibration

# Build stereo calibration tool
cd stereo_calibration
mkdir build && cd build
cmake ..
make -j$(nproc)

# Build reprojection viewer
cd ../../reprojection_viewer
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

### Stereo Calibration

The calibration process requires a printed checkerboard pattern.

Default configuration expects a **9x6 checkerboard with 24mm squares**. For other configurations, please update the checkerboard description in the code directly.


#### Prepare Calibration Target
- Print a checkerboard pattern on rigid, flat surface
- Default: 9 horizontal × 6 vertical inner corners, 24mm square size
- Ensure the pattern is perfectly flat and well-lit

#### Run Calibration
```bash
cd stereo_calibration/build
./ZED_Opencv_Calibration [options]
```

**Options:**
- `--svo <file>` - Use SVO file instead of live camera
- `--fisheye` - Use fisheye distortion model (default: RadTan)
- `--help` - Show help message

#### Calibration Process
1. **Coverage Phase**: Move the checkerboard to cover different areas of the image
   - Green areas indicate sufficient coverage
   - Press 's' to capture frames when checkerboard is detected
   
2. **Rotation Phase**: Rotate the checkerboard in different orientations
   - Achieve >60° rotation in X, Y axes and >60° distance variation
   - Avoid rotating >45° around Z-axis
   - Progress indicators show completion percentage

3. **Completion**: Calibration runs automatically when sufficient data is collected
   - Minimum 15 images required
   - Calibration file saved as `zed_calibration_[serial].yml`

### Reprojection Viewer

Visualize the rectification of the image on the full 

```bash
cd reprojection_viewer/build
./ZED_Depth_Repro [options]
```

## Configuration

### Calibration Parameters
Edit the following constants in `stereo_calibration/src/main.cpp`:

```cpp
constexpr int target_w = 9;        // Horizontal inner corners
constexpr int target_h = 6;        // Vertical inner corners  
constexpr float square_size = 24.0; // Square size in mm
```

### Quality Thresholds
```cpp
const float min_coverage = 10;      // Coverage percentage
const float min_rotation = 60;      // Rotation in degrees
const float min_distance = 200;     // Distance variation in mm
```

## Advanced Features

### Virtual Stereo Camera Support
Configure multiple ZED One cameras as a virtual stereo pair:

```cpp
// In stereo_calibration/src/main.cpp
if(1){ // Enable virtual stereo
    const int sn_left = 300000001;   // Serial number of left camera
    const int sn_right = 300000002;  // Serial number of right camera
    int sn_stereo = sl::generateVirtualStereoSerialNumber(sn_left, sn_right);
    init_params.input.setVirtualStereoFromSerialNumbers(sn_left, sn_right, sn_stereo);
}
```

### Fisheye Camera Support

For wide-angle or fisheye lenses:
```bash
./ZED_Opencv_Calibration --fisheye
```