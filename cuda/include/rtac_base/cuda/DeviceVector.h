#ifndef _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <rtac_base/types/SharedVector.h>

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

template <typename T>
class HostVector;

template <typename T>
class DeviceVector
{
    public:

    using value_type      = T;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    protected:

    T*     data_;
    size_t size_;
    size_t capacity_;

    void allocate(size_t size);
    void free();

    public:

    DeviceVector();
    DeviceVector(size_t size);
    DeviceVector(const DeviceVector<T>& other);
    DeviceVector(const HostVector<T>& other);
    DeviceVector(const std::vector<T>& other);
    ~DeviceVector();

    void copy_from_host(size_t size, const T* data);
    
    DeviceVector& operator=(const DeviceVector<T>& other);
    DeviceVector& operator=(const HostVector<T>& other);
    DeviceVector& operator=(const std::vector<T>& other);

    void resize(size_t size);
    size_t size() const;
    size_t capacity() const;

    pointer       data();
    const_pointer data() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
};
template <typename T>
using SharedDeviceVector = rtac::types::SharedVectorBase<DeviceVector<T>>;

// implementation
template <typename T>
DeviceVector<T>::DeviceVector() :
    data_(NULL),
    size_(0),
    capacity_(0)
{}

template <typename T>
DeviceVector<T>::DeviceVector(size_t size) :
    DeviceVector()
{
    this->resize(size);
}

template <typename T>
DeviceVector<T>::DeviceVector(const DeviceVector<T>& other) :
    DeviceVector(other.size())
{
    *this = other;
}

template <typename T>
DeviceVector<T>::DeviceVector(const HostVector<T>& other) :
    DeviceVector(other.size())
{
    *this = other;
}

template <typename T>
DeviceVector<T>::DeviceVector(const std::vector<T>& other) :
    DeviceVector(other.size())
{
    *this = other;
}

template <typename T>
DeviceVector<T>::~DeviceVector()
{
    this->free();
}

template <typename T>
void DeviceVector<T>::copy_from_host(size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyHostToDevice) );
}

template <typename T>
DeviceVector<T>& DeviceVector<T>::operator=(const DeviceVector<T>& other)
{
    this->resize(other.size());
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(other.data_),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToDevice) );
    return *this;
}

template <typename T>
DeviceVector<T>& DeviceVector<T>::operator=(const HostVector<T>& other)
{
    this->resize(other.size());
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(other.data()),
                           sizeof(T)*size_,
                           cudaMemcpyHostToDevice) );
    return *this;
}

template <typename T>
DeviceVector<T>& DeviceVector<T>::operator=(const std::vector<T>& other)
{
    this->resize(other.size());
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(other.data()),
                           sizeof(T)*size_,
                           cudaMemcpyHostToDevice) );
    return *this;
}

template <typename T>
void DeviceVector<T>::allocate(size_t size)
{
    this->free();
    CUDA_CHECK( cudaMalloc(&data_, sizeof(T)*size) );
    capacity_ = size;
}

template <typename T>
void DeviceVector<T>::free()
{
    CUDA_CHECK( cudaFree(data_) );
    capacity_ = 0;
    size_     = 0;
}

template <typename T>
void DeviceVector<T>::resize(size_t size)
{
    if(capacity_ < size)
        this->allocate(size);
    size_ = size;
}

template <typename T>
size_t DeviceVector<T>::size() const
{
    return size_;
}

template <typename T>
size_t DeviceVector<T>::capacity() const
{
    return capacity_;
}

template <typename T> typename DeviceVector<T>::
pointer DeviceVector<T>::data()
{
    return data_;
}

template <typename T> typename DeviceVector<T>::
const_pointer DeviceVector<T>::data() const
{
    return data_;
}

template <typename T> typename DeviceVector<T>::
iterator DeviceVector<T>::begin()
{
    return data_;
}

template <typename T> typename DeviceVector<T>::
iterator DeviceVector<T>::end()
{
    return data_ + size_;
}

template <typename T> typename DeviceVector<T>::
const_iterator DeviceVector<T>::begin() const
{
    return data_;
}

template <typename T> typename DeviceVector<T>::
const_iterator DeviceVector<T>::end() const
{
    return data_ + size_;
}

}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::DeviceVector<T>& v)
{
    os << rtac::cuda::HostVector<T>(v);
    return os;
}

#endif //_DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
