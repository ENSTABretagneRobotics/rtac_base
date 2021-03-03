#ifndef _DEF_RTAC_BASE_CUDA_DEVICE_OBJECT_H_
#define _DEF_RTAC_BASE_CUDA_DEVICE_OBJECT_H_

#include <cuda_runtime.h>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

template <class T>
class DeviceObjectBase
{
    public:

    using value_type = T;

    virtual operator T&() = 0;
    virtual operator const T&() const = 0;

    virtual T* host_ptr() = 0;
    virtual const T* host_ptr() const = 0;

    virtual T* device_ptr() = 0;
    virtual const T* device_ptr() const = 0;

    void update_device(); // const ?
    void update_host(); // const ?

    void update_device(cudaStream_t stream); // const ?
    void update_host(cudaStream_t stream); // const ?
};

template <class T>
class DeviceObjectPtr : public T, public DeviceObjectBase<T>
{
    public:

    using value_type = T;

    protected:

    T* devicePtr_;

    public:

    DeviceObjectPtr(T* devicePtr) : 
        devicePtr_(devicePtr)
    {}

    DeviceObjectPtr(const T& other, T* devicePtr) : 
        T(other), devicePtr_(devicePtr)
    {}

    virtual operator T&() { return *this; }
    virtual operator const T&() const { return *this; }

    virtual T* host_ptr() { return this; }
    virtual const T* host_ptr() const { return this; }

    virtual T* device_ptr() { return devicePtr_; }
    virtual const T* device_ptr() const { return devicePtr_; }
};

template <class T>
class DeviceObject : public DeviceObjectPtr<T>
{
    public:

    using value_type = T;

    private:

    void allocate() {
        CUDA_CHECK( cudaMalloc(&this->devicePtr_, sizeof(T)) );
    }

    public:

    DeviceObject() : DeviceObjectPtr<T>(nullptr) {
        this->allocate();
    }

    DeviceObject(const T& other) : DeviceObjectPtr<T>(other, nullptr) {
        this->allocate();
    }

    ~DeviceObject() {
        CUDA_CHECK( cudaFree(this->devicePtr_) );
    }
};

// Implementations

// DeviceObjectBase //////////////////////////////////////

template <class T>
void DeviceObjectBase<T>::update_device()
{
    CUDA_CHECK( cudaMemcpy(this->device_ptr(), this->host_ptr(),
                           sizeof(T), cudaMemcpyHostToDevice) );
}

template <class T>
void DeviceObjectBase<T>::update_host()
{
    CUDA_CHECK( cudaMemcpy(this->host_ptr(), this->device_ptr(),
                           sizeof(T), cudaMemcpyDeviceToHost) );
}

template <class T>
void DeviceObjectBase<T>::update_device(cudaStream_t stream)
{
    CUDA_CHECK( cudaMemcpy(this->device_ptr(), this->host_ptr(),
                           sizeof(T), cudaMemcpyHostToDevice, stream) );
}

template <class T>
void DeviceObjectBase<T>::update_host(cudaStream_t stream)
{
    CUDA_CHECK( cudaMemcpy(this->host_ptr(), this->device_ptr(),
                           sizeof(T), cudaMemcpyDeviceToHost, stream) );
}

}; //namespace cuda
}; //namespace rtac


#endif //_DEF_RTAC_BASE_CUDA_DEVICE_OBJECT_H_
