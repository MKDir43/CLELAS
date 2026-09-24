# CLELAS

CLELAS is an OpenCL implementation of the ELAS (Efficient Large-Scale Stereo Matching) algorithm. It computes a disparity map from a rectified stereo image pair. Both a CPU reference implementation and an OpenCL implementation are provided.

## Features

- Large-scale stereo matching based on the ELAS algorithm
- Runs on any OpenCL 2.0+ platform (the kernels are written in OpenCL C 1.2)
- Numerically consistent with the CPU reference implementation
- OAK-D stereo camera input support (when depthai is available)

## Directory Structure

```
CLELAS/
├── cpu/                    # CPU reference implementation (library)
├── opencl/                 # OpenCL implementation (library)
│   ├── src/                # Host-side processing
│   └── kernels/            # OpenCL kernels (.cl) and host-side wrappers
├── sample/                 # Sample programs
├── test/                   # Test code
├── generate_test_images.py # Generates a synthetic stereo pair
├── CMakeLists.txt
└── LICENSE
```

## Dependencies

The following are required to build the project.

- CMake 3.10 or later
- OpenCV 3.4 or later
- An OpenCL 2.0+ platform (ICD loader and headers)
- OpenCL C++ bindings `CL/opencl.hpp` (Khronos OpenCL-CLHPP; `opencl-clhpp-headers` on Ubuntu)
- Boost (program_options, system, filesystem; for the samples)
- Google Test (for the tests)
- depthai (optional; for the OAK-D sample)

Any OpenCL runtime can be used, such as a GPU vendor runtime (NVIDIA, Intel, AMD) or PoCL.

## Build

```bash
cd CLELAS
mkdir build && cd build
cmake -DBUILD_SAMPLE=ON ..
make -j4
```

To include the tests:

```bash
cmake -DBUILD_SAMPLE=ON -DBUILD_TESTS=ON ..
make -j4
```

Build options:

- `BUILD_SAMPLE` (default ON): builds the sample programs
- `BUILD_TESTS` (default OFF): builds the test suite
- `CMAKE_BUILD_TYPE` (default `Release`): use `Debug` for a debug build

## Running

The commands below are run from the `build` directory created above. The OpenCL
library looks for its kernels in `kernels/` under the working directory first,
then in the directory `make` copied them to, so the samples also work from other
directories.

### Sample images

Generate a synthetic stereo pair (shapes drawn at known disparities):

```bash
python3 ../generate_test_images.py
```

This writes `input_left.png` / `input_right.png` for the CPU sample, and
`left/` / `right/` directories for the OpenCL sample.

### CPU version

`image_cpu` takes the left/right images and the output paths as files:

```bash
./sample/image_cpu input_left.png input_right.png output_left.png output_right.png
```

### OpenCL version

`image_opencl` processes a directory of frames; the left and right frames must
share the same filename. The output directories are created automatically:

```bash
./sample/image_opencl --inputDirectoryLeft left --inputDirectoryRight right \
                      --outputDirectoryLeft out_left --outputDirectoryRight out_right \
                      --sizeWidth 1280 --sizeHeight 640
```

### OAK-D version (requires depthai)

```bash
./sample/oakd_opencl
```

## Testing

The test suite validates accuracy against the KITTI dataset. Place the images
under `data/testimages/kitti/` (left, right, and ground-truth disparity), then:

```bash
cd build
ctest --verbose
```

## Algorithm Pipeline

Disparity is computed in the following order.

1. Sobel filtering for edge detection
2. Support point extraction
3. Delaunay triangulation
4. Per-triangle disparity plane estimation
5. Grid interpolation
6. Left-right consistency check
7. Post-processing as in libelas: small segment removal, gap interpolation, and the
   optional adaptive mean and median filters

## Input / Output

- Input: rectified stereo image pairs (8-bit grayscale or color)
- Output: disparity maps (32-bit float, pixel units); with `subsampling`, only every second
  pixel is computed and the maps are width/2 x height/2, as in libelas
- Image format: PNG via OpenCV

## License

CLELAS is licensed under the GNU General Public License v3.0 (see [LICENSE](LICENSE)).
It is derived from [libelas](https://www.cvlibs.net/software/libelas/) by Andreas Geiger
(Copyright 2011, Institute of Measurement and Control Systems, Karlsruhe Institute of
Technology), which is licensed under the GPL v3 as well.

`triangle.cpp` / `triangle.h` (in `cpu/src/` and `opencl/src/`) are Triangle by Jonathan
Richard Shewchuk, modified by Andreas Geiger, and keep their own license, stated at the top of
the files: they may be redistributed free of charge with the notices kept, and distributing
them as part of a commercial system requires an arrangement with the author.
