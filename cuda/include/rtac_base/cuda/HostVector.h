#ifndef _DEF_RTAC_BASE_CUDA_HOST_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_HOST_VECTOR_H_

#include <iostream>
#include <vector>

#include <rtac_base/types/SharedVector.h>

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

template <typename T>
class DeviceVector;

template <typename T>
class HostVector
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

    HostVector();
    HostVector(size_t size);
    HostVector(const HostVector<T>& other);
    HostVector(const DeviceVector<T>& other);
    HostVector(const std::vector<T>& other);
    ~HostVector();
    
    HostVector& operator=(const HostVector<T>& other);
    HostVector& operator=(const DeviceVector<T>& other);
    HostVector& operator=(const std::vector<T>& other);

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
using SharedHostVector = rtac::types::SharedVectorBase<HostVector<T>>;

// implementation
template <typename T>
HostVector<T>::HostVector() :
    data_(NULL),
    size_(0),
    capacity_(0)
{}

template <typename T>
HostVector<T>::HostVector(size_t size) :
    HostVector()
{
    this->resize(size);
}

template <typename T>
HostVector<T>::HostVector(const HostVector<T>& other) :
    HostVector(other.size())
{
    //cuda::memcpy::host_to_host(data_, other.data(), size_);
    *this = other;
}

template <typename T>
HostVector<T>::HostVector(const DeviceVector<T>& other) :
    HostVector(other.size())
{
    //cuda::memcpy::device_to_host(data_, other.data(), size_);
    *this = other;
}

template <typename T>
HostVector<T>::HostVector(const std::vector<T>& other) :
    HostVector(other.size())
{
    //cuda::memcpy::host_to_host(data_, other.data(), size_);
    *this = other;
}

template <typename T>
HostVector<T>::~HostVector()
{
    this->free();
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const HostVector<T>& other)
{
    this->resize(other.size());
    // use std::memcpy instead ?
    cuda::memcpy::host_to_host(data_, other.data(), size_);
    return *this;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const DeviceVector<T>& other)
{
    this->resize(other.size());
    cuda::memcpy::device_to_host(data_, other.data(), size_);
    return *this;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const std::vector<T>& other)
{
    this->resize(other.size());
    // use std::memcpy instead ?
    cuda::memcpy::host_to_host(data_, other.data(), size_);
    return *this;
}

template <typename T>
void HostVector<T>::allocate(size_t size)
{
    this->free();
    data_ = new T[size];
    capacity_ = size;
}

template <typename T>
void HostVector<T>::free()
{
    delete[] data_;
    capacity_ = 0;
    size_     = 0;
}

template <typename T>
void HostVector<T>::resize(size_t size)
{
    if(capacity_ < size)
        this->allocate(size);
    size_ = size;
}

template <typename T>
size_t HostVector<T>::size() const
{
    return size_;
}

template <typename T>
size_t HostVector<T>::capacity() const
{
    return capacity_;
}

template <typename T> typename HostVector<T>::
pointer HostVector<T>::data()
{
    return data_;
}

template <typename T> typename HostVector<T>::
const_pointer HostVector<T>::data() const
{
    return data_;
}

template <typename T> typename HostVector<T>::
iterator HostVector<T>::begin()
{
    return data_;
}

template <typename T> typename HostVector<T>::
iterator HostVector<T>::end()
{
    return data_ + size_;
}

template <typename T> typename HostVector<T>::
const_iterator HostVector<T>::begin() const
{
    return data_;
}

template <typename T> typename HostVector<T>::
const_iterator HostVector<T>::end() const
{
    return data_ + size_;
}

}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::HostVector<T>& v)
{
    os << "(";
    auto it = v.begin();
    if(v.size() <= 16) {
        os << *it;
        it++;
        for(; it != v.end(); it++) {
            os << " " << *it;
        }
    }
    else {
        for(auto it = v.begin(); it != v.begin() + 3; it++) {
            os << *it << " ";
        }
        os << "...";
        for(auto it = v.end() - 3; it != v.end(); it++) {
            os << " " << *it;
        }
    }
    os << ")";
    return os;
}

#endif //_DEF_RTAC_BASE_CUDA_HOST_VECTOR_H_
