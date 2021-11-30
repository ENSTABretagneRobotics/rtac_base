#ifndef _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/types/SharedVector.h>

#ifndef RTAC_CUDACC
#include <rtac_base/types/common.h>
#endif

#ifdef RTAC_CUDACC
#include <thrust/device_ptr.h>
#endif


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
    
    #ifndef RTAC_CUDACC
    DeviceVector(const types::Vector<T>& other);
    DeviceVector& operator=(const types::Vector<T>& other);
    #endif

    void resize(size_t size);
    size_t size() const;
    size_t capacity() const;

    pointer       data();
    const_pointer data() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    #ifdef RTAC_CUDACC  // the following methods are only usable in CUDA code.
    value_type& operator[](size_t idx);
    const value_type& operator[](size_t idx) const;

    value_type& front();
    const value_type& front() const;
    value_type& back();
    const value_type& back() const;

    thrust::device_ptr<T>       begin_thrust();
    thrust::device_ptr<T>       end_thrust();
    thrust::device_ptr<const T> begin_thrust() const;
    thrust::device_ptr<const T> end_thrust() const;
    #endif
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

#ifndef RTAC_CUDACC
template <typename T>
DeviceVector<T>::DeviceVector(const types::Vector<T>& other) :
    DeviceVector(other.size())
{
    *this = other;
}

template <typename T>
DeviceVector<T>& DeviceVector<T>::operator=(const types::Vector<T>& other)
{
    this->resize(other.size());
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(other.data()),
                           sizeof(T)*size_,
                           cudaMemcpyHostToDevice) );
    return *this;
}
#endif

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

#ifdef RTAC_CUDACC
template <typename T> typename DeviceVector<T>::
value_type& DeviceVector<T>::operator[](size_t idx)
{
    return data_[idx];
}

template <typename T> const typename DeviceVector<T>::
value_type& DeviceVector<T>::operator[](size_t idx) const
{
    return data_[idx];
}

template <typename T> typename DeviceVector<T>::
value_type& DeviceVector<T>::front()
{
    return data_[0];
}

template <typename T> const typename DeviceVector<T>::
value_type& DeviceVector<T>::front() const
{
    return data_[0];
}

template <typename T> typename DeviceVector<T>::
value_type& DeviceVector<T>::back()
{
    return data_[this->size() - 1];
}

template <typename T> const typename DeviceVector<T>::
value_type& DeviceVector<T>::back() const
{
    return data_[this->size() - 1];
}

template <typename T>
thrust::device_ptr<T> DeviceVector<T>::begin_thrust()
{
    return thrust::device_pointer_cast(data_);
}

template <typename T>
thrust::device_ptr<T> DeviceVector<T>::end_thrust()
{
    return thrust::device_pointer_cast(data_ + size_);
}

template <typename T>
thrust::device_ptr<const T> DeviceVector<T>::begin_thrust() const
{
    return thrust::device_pointer_cast(data_);
}

template <typename T>
thrust::device_ptr<const T> DeviceVector<T>::end_thrust() const
{
    return thrust::device_pointer_cast(data_ + size_);
}
#endif //RTAC_CUDACC

}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::DeviceVector<T>& v)
{
    os << rtac::cuda::HostVector<T>(v);
    return os;
}

#endif //_DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
