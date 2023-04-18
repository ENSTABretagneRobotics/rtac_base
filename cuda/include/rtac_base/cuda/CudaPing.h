#ifndef _DEF_RTAC_BASE_CUDA_CUDA_PING_H_
#define _DEF_RTAC_BASE_CUDA_CUDA_PING_H_

#include <rtac_base/types/SonarPing.h>
#include <rtac_base/cuda/CudaVector.h>

namespace rtac { namespace cuda {

template <typename T> using CudaPing2D = rtac::Ping2D<T,CudaVector>;

} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_CUDA_PING_H_
