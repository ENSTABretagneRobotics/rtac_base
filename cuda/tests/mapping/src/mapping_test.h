#ifndef _DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_
#define _DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/Mapping.h>
using namespace rtac::cuda;

// This Functor transforms pixel coordinates in normalized texture coordinates,
// given a size (width, height).
struct NormalizerUV {
    using InputT  = uint2;
    using OutputT = float2;

    uint2 shape;

    RTAC_HOSTDEVICE float2 operator()(uint2 pixCoords) const {
        return float2({(2.0f* pixCoords.x) / (shape.x - 1),
                       (2.0f* pixCoords.y) / (shape.y - 1)});
    }
};

using Mapping1 = Mapping<float>;
using Mapping2 = Mapping<float, NormalizerUV>;

DeviceVector<float> render_texture(int W, int H, const Texture2D<float>& texData);

DeviceVector<float> render_mapping1(int W, int H, const Mapping1::DeviceMap& map);
DeviceVector<float> render_mapping2(int W, int H, const Mapping2::DeviceMap& map);

#endif //_DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_
