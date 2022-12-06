#ifndef _DEF_RTAC_BASE_CUDA_DEVICE_REFERENCE_H_
#define _DEF_RTAC_BASE_CUDA_DEVICE_REFERENCE_H_

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace rtac { namespace cuda {

/**
 * This is an helper to be able to easily modify a host side object with a
 * CUDA kernel call.
 *
 * It work by copying the content of target to the device on construction, and
 * copying back the result into the target on destruction. It allows for this
 * kind of construct :
 *
 *      __global__ void increment(float* v) { ++(*v); }
 *
 *      float value = 1;
 *      increment<<<1,1>>>(Ref(value));
 *      // value is now 2; 
 *
 * This is mostly to be used for testing as it is very inefficient.
 */
template <class T>
struct Ref
{
    protected:

    T&           target_;
    T*           deviceData_;
    cudaStream_t stream_;       // not used for now

    public:

    Ref(T& target, cudaStream_t stream = 0) :
        target_(target), deviceData_(nullptr), stream_(stream)
    {
        CUDA_CHECK( cudaMalloc(&deviceData_, sizeof(T)) );
        CUDA_CHECK( cudaMemcpy(deviceData_, &target_, sizeof(T),
                               cudaMemcpyHostToDevice) );
    }

    ~Ref() {
        cudaMemcpy(&target_, deviceData_, sizeof(T),
                   cudaMemcpyDeviceToHost);
        cudaFree(deviceData_);
    }

    // Implicitly castable to pointer.
    operator       T*()       { return deviceData_; }
    operator const T*() const { return deviceData_; }

    // Have to put this because of optixLaunch param API
          T* get()       { return deviceData_; }
    const T* get() const { return deviceData_; }
};


// CUDA 11.2+ only.
//template <class T>
//struct Ref
//{
//    protected:
//
//    T&           target_;
//    T*           deviceData_;
//    cudaStream_t stream_;
//
//    public:
//
//    Ref(T& target, cudaStream_t stream = 0) :
//        target_(target), deviceData_(nullptr), stream_(stream)
//    {
//        CUDA_CHECK( cudaMallocAsync(&deviceData_, sizeof(T), stream_) );
//        CUDA_CHECK( cudaMemcpyAsync(deviceData_, &target_, sizeof(T),
//                                    cudaMemcpyHostToDevice, stream_) );
//    }
//
//    ~Ref() {
//        cudaMemcpyAsync(&target_, deviceData_, sizeof(T),
//                        cudaMemcpyDeviceToHost, stream_);
//        cudaFreeAsync(deviceData_);
//    }
//
//    // Implicitly castable to pointer.
//    operator       T*()       { return deviceData_; }
//    operator const T*() const { return deviceData_; }
//};

} //namespace cuda
} //namespace rtac



#endif //_DEF_RTAC_BASE_CUDA_DEVICE_REFERENCE_H_
