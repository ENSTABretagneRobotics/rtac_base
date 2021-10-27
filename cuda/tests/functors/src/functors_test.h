#ifndef _DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_
#define _DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/FunctorCompound.h>
#include <rtac_base/cuda/functors.h>

namespace rtac { namespace cuda {


struct Vectorize4 {
    using InputT  = float;
    using OutputT = float4;

    float x;

    RTAC_HOSTDEVICE float4 operator()(float input) const {
        return float4({input, input, input, input});
    }
};

struct Norm4 {
    using InputT  = float4;
    using OutputT = float;

    float x;

    RTAC_HOSTDEVICE float operator()(const float4& input) const {
        // return length(input); // WHY U NOT WORKING ???
        return sqrt( input.x*input.x
                   + input.y*input.y
                   + input.z*input.z
                   + input.w*input.w);
                    
    }
};

using MultiType = functors::FunctorCompound<Norm4, Vectorize4>;

using Saxpy = functors::FunctorCompound<functors::Offset<float>, functors::Scaling<float>>;


DeviceVector<float> scaling(const DeviceVector<float>& input, 
                            const functors::Scaling<float>& func);
DeviceVector<float> saxpy(const DeviceVector<float>& input, const Saxpy& func);

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_
