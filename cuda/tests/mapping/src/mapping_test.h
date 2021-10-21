#ifndef _DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_
#define _DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_

#include <rtac_base/cuda/Mapping.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

struct Affine1D
{
    using Input = float;

    float a;
    float b;

    #ifdef RTAC_CUDACC
    __device__ float2 operator()(float x) const {
        return float2({a*x+b, 0.0});
    }
    #endif //RTAC_CUDACC
};

DeviceVector<float> map(const DeviceMapping<Affine1D,float>& mapping,
                        const DeviceVector<float>& x);

#endif //_DEF_RTAC_BASE_CUDA_TESTS_MAPPING_TEST_H_
