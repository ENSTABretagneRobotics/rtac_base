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

/**
 * This act in host code as a const reference to an element in cuda global
 * memory.
 *
 * This is a very inefficient way of accessing device data. Use with care.
 */
template <typename T>
class ConstGlobalRef
{
    protected:

    const T* ptr_; // pointer to element in cuda global memory

    public:
    
    ConstGlobalRef() = delete;
    ConstGlobalRef(const T* ptr) : ptr_(ptr) {}

    operator T() const {
        T res;
        cudaMemcpy(&res, ptr_, sizeof(T), cudaMemcpyDeviceToHost);
        return res;
    }
    const T* ptr()       const { return ptr_; }
    const T* operator&() const { return ptr_; }
};


/**
 * This act in host code as a mutable reference to an element in cuda global
 * memory.
 *
 * This is a very inefficient way of accessing device data. Use with care.
 */
template <typename T>
class GlobalRef : public ConstGlobalRef<T>
{
    public:

    GlobalRef() = delete;
    GlobalRef(const GlobalRef<T>&) = default;

    GlobalRef(T* ptr) : ConstGlobalRef<T>(ptr) {}
    GlobalRef(const ConstGlobalRef<T>& other) : ConstGlobalRef<T>(other) {}

    T* ptr()       { return const_cast<T*>(this->ptr_); }
    T* operator&() { return const_cast<T*>(this->ptr_); }

    GlobalRef& operator=(T other) {
        cudaMemcpy(this->ptr(), &other, sizeof(T), cudaMemcpyHostToDevice);
        return *this;
    }
    GlobalRef& operator=(const ConstGlobalRef<T>& other) {
        cudaMemcpy(this->ptr(), other.ptr(), sizeof(T), cudaMemcpyDeviceToDevice);
        return *this;
    }
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
