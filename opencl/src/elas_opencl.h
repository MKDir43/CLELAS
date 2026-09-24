#ifndef CLELAS_PROCESSOR_H
#define CLELAS_PROCESSOR_H

#include "elas_parameter.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace clelas {

// Defined in elas_resource.h (internal).
struct HostResources;
struct HostSupportPoints;
struct HostTriangleCorners;
struct OpenCLResources;

// ELAS stereo matching accelerated with OpenCL.
// OpenCL failures are reported as std::runtime_error.
class Processor
{
public:
    Processor(int32_t width, int32_t height, int32_t pitch,
              const Parameters& param, bool estimates_subpixel);
    ~Processor();

    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;

    // Computes the left and right disparity maps (float[pitch * height] each)
    // from 8-bit grayscale images (uint8_t[pitch * height] each).
    // With param.subsampling, only every 2nd pixel is computed and the maps are
    // width/2 x height/2 with rows width/2 apart, as in the CPU version.
    // Returns 0 on success, or -1 when too few support points or triangles are found.
    int32_t process(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right);

    int32_t getWidth() const { return width; }
    int32_t getHeight() const { return height; }
    int32_t getPitch() const { return pitch; }

    // Saves the Sobel images of the last processed frame as <prefix>_*.png.
    void saveIntermediateImages(const char* prefix);

#ifdef PROFILE
public:
#else
private:
#endif
    int32_t width;
    int32_t height;
    int32_t pitch;
    Parameters param;
    bool estimates_subpixel;
    std::unique_ptr<HostResources> host_;
    std::unique_ptr<OpenCLResources> opencl_;

    int32_t processFrame(
        uint8_t* image_left, uint8_t* image_right,
        float* disp_left, float* disp_right);

    // Correct Support Point Candidates.
    void removeInconsistentSupportPoints(
        int16_t* D_can, int32_t D_can_width, int32_t D_can_height);
    void removeRedundantSupportPoints(
        int16_t* D_can, int32_t D_can_width, int32_t D_can_height,
        int32_t redun_max_dist, int32_t redun_threshold, bool vertical);
    void correctSupportPointCandidates(
        int16_t* candidates, int32_t candi_width, int32_t candi_height);
    int32_t addCornerSupportPoints(
        int32_t* sup_us, int32_t* sup_vs, int32_t* sup_ds, int32_t sup_num);
    int32_t convertCandidatesToSupportPoints(
        const int16_t* candidates, int32_t candi_width, int32_t candi_height, int32_t candi_step,
        HostSupportPoints& supports);

    // Delaunay Triangulation
    int32_t computeDelaunayTriangulation(
        const HostSupportPoints& supports, size_t sup_num,
        HostTriangleCorners& tri, size_t triangle_max,
        bool right_image);

    // Post-processing (host part)
    void removeSmallSegments(float* D);
    void gapInterpolation(float* D);
};

}
#endif // CLELAS_PROCESSOR_H
