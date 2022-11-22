#ifndef _DEF_RTAC_BASE_CUDA_HOST_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_HOST_VECTOR_H_

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/types/VectorView.h>

#ifndef RTAC_CUDACC
#include <rtac_base/types/common.h>
#endif

namespace rtac { namespace cuda {

template <typename T>
class DeviceVector;
template <typename T>
class PinnedVector;

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
    HostVector(const PinnedVector<T>& other);
    HostVector(const std::vector<T>& other);
    ~HostVector();
    
    void copy_from_host(size_t size, const T* data);
    void copy_from_device(size_t size, const T* data);

    HostVector& operator=(const HostVector<T>& other);
    HostVector& operator=(const DeviceVector<T>& other);
    HostVector& operator=(const PinnedVector<T>& other);
    HostVector& operator=(const std::vector<T>& other);

    #ifndef RTAC_CUDACC
    HostVector(const rtac::Vector<T>& other);
    HostVector& operator=(const rtac::Vector<T>& other);
    #endif

    void resize(size_t size);
    void clear() { this->free(); }
    size_t size() const;
    size_t capacity() const;

    pointer       data();
    const_pointer data() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    auto view() const { return rtac::make_view(*this); }
    auto view()       { return rtac::make_view(*this); }

    value_type& operator[](size_t idx);
    const value_type& operator[](size_t idx) const;

    value_type& front();
    const value_type& front() const;
    value_type& back();
    const value_type& back() const;
};

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
    *this = other;
}

template <typename T>
HostVector<T>::HostVector(const DeviceVector<T>& other) :
    HostVector(other.size())
{
    *this = other;
}

template <typename T>
HostVector<T>::HostVector(const PinnedVector<T>& other) :
    HostVector(other.size())
{
    *this = other;
}

template <typename T>
HostVector<T>::HostVector(const std::vector<T>& other) :
    HostVector(other.size())
{
    *this = other;
}

template <typename T>
HostVector<T>::~HostVector()
{
    this->free();
}

template <typename T>
void HostVector<T>::copy_from_host(size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyHostToHost) );
}

template <typename T>
void HostVector<T>::copy_from_device(size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToHost) );
}
template <typename T>
HostVector<T>& HostVector<T>::operator=(const HostVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const DeviceVector<T>& other)
{
    this->copy_from_device(other.size(), other.data());
    return *this;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const PinnedVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const std::vector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

#ifndef RTAC_CUDACC
template <typename T>
HostVector<T>::HostVector(const rtac::Vector<T>& other) :
    HostVector(other.size())
{
    *this = other;
}

template <typename T>
HostVector<T>& HostVector<T>::operator=(const rtac::Vector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}
#endif

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

template <typename T> typename HostVector<T>::
value_type& HostVector<T>::operator[](size_t idx)
{
    return data_[idx];
}

template <typename T> const typename HostVector<T>::
value_type& HostVector<T>::operator[](size_t idx) const
{
    return data_[idx];
}

template <typename T> typename HostVector<T>::
value_type& HostVector<T>::front()
{
    return data_[0];
}

template <typename T> const typename HostVector<T>::
value_type& HostVector<T>::front() const
{
    return data_[0];
}

template <typename T> typename HostVector<T>::
value_type& HostVector<T>::back()
{
    return data_[this->size() - 1];
}

template <typename T> const typename HostVector<T>::
value_type& HostVector<T>::back() const
{
    return data_[this->size() - 1];
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
