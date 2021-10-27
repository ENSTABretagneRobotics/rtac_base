#ifndef _DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_
#define _DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/FunctorCompound.h>
#include <rtac_base/cuda/functors.h>

namespace rtac { namespace cuda {

using Saxpy = functors::FunctorCompound<functors::Offset<float>, functors::Scaling<float>>;

DeviceVector<float> scaling(const DeviceVector<float>& input, 
                            const functors::Scaling<float>& func);
DeviceVector<float> saxpy(const DeviceVector<float>& input, const Saxpy& func);

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_TESTS_FUNCTORS_TEST_H_
