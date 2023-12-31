#ifndef _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_

#include <iostream>
#include <vector>
#include <utility>

#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceReference.h>

namespace rtac { namespace display {
    template <typename T> class GLVector;
}}

namespace rtac { namespace cuda {

template <typename T>
class PinnedVector;

template <typename T>
class CudaVector
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

    static CudaVector<T> linspace(T first, T last, std::size_t size) {
        return HostVector<T>::linspace(first, last, size);
    }

    protected:

    T*          data_;
    std::size_t size_;
    std::size_t capacity_;

    void allocate(std::size_t size);
    void free();

    public:

    CudaVector();
    CudaVector(std::size_t size);
    CudaVector(const CudaVector<T>& other);
    CudaVector(const HostVector<T>& other);
    CudaVector(const PinnedVector<T>& other);
    CudaVector(const std::vector<T>& other);
    ~CudaVector();

    //CudaVector(CudaVector<T>&& other) : CudaVector() { *this = std::move(other); }
    //CudaVector<T>& operator=(CudaVector<T>&& other) {
    //    std::exchange(data_,     other.data_);
    //    std::exchange(size_,     other.size_);
    //    std::exchange(capacity_, other.capacity_);
    //    return *this;
    //}


    void copy_from_host(std::size_t size, const T* data);
    void copy_to_host(T* dst) const;
    void copy_from_cuda(std::size_t size, const T* data);
    void copy_to_cuda(T* dst) const;

    [[deprecated]]
    void copy_from_device(std::size_t size, const T* data);
    
    CudaVector& operator=(const CudaVector<T>& other);
    CudaVector& operator=(const HostVector<T>& other);
    CudaVector& operator=(const PinnedVector<T>& other);
    CudaVector& operator=(const std::vector<T>& other);

    void resize(std::size_t size);
    void clear() { this->free(); }

    std::size_t size()     const { return size_;     }
    std::size_t capacity() const { return capacity_; }

    const T* data() const { return data_; }
          T* data()       { return data_; }

    const T* cbegin() const { return data_; }
    const T* begin()  const { return data_; }
          T* begin()        { return data_; }
    const T* cend() const   { return data_ + size_; }
    const T* end()  const   { return data_ + size_; }
          T* end()          { return data_ + size_; }

    GlobalRef<T>      operator[](std::size_t idx);
    ConstGlobalRef<T> operator[](std::size_t idx) const;
    GlobalRef<T>      front();
    GlobalRef<T>      back();
    ConstGlobalRef<T> front() const;
    ConstGlobalRef<T> back()  const;

    auto const_view() const { return this->view();                                    }
    //auto view()       const { return VectorView<const T>(this->size(), this->data()); }
    auto view()       const { return ConstVectorView<T>(this->size(), this->data());  }
    auto view()             { return VectorView<T>(this->size(), this->data());       }

    CudaVector(const display::GLVector<T>& other) { *this = other; }
    CudaVector& operator=(const display::GLVector<T>& other) {
        this->resize(other.size());
        other.copy_to_cuda(this->data());
        return *this;
    }
};

// implementation
template <typename T> inline
CudaVector<T>::CudaVector() :
    data_(NULL),
    size_(0),
    capacity_(0)
{}

template <typename T> inline
CudaVector<T>::CudaVector(std::size_t size) :
    CudaVector()
{
    this->resize(size);
}

template <typename T> inline
CudaVector<T>::CudaVector(const CudaVector<T>& other) :
    CudaVector(other.size())
{
    *this = other;
}

template <typename T> inline
CudaVector<T>::CudaVector(const HostVector<T>& other) :
    CudaVector(other.size())
{
    *this = other;
}

template <typename T> inline
CudaVector<T>::CudaVector(const PinnedVector<T>& other) :
    CudaVector(other.size())
{
    *this = other;
}

template <typename T> inline
CudaVector<T>::CudaVector(const std::vector<T>& other) :
    CudaVector(other.size())
{
    *this = other;
}

template <typename T> inline
CudaVector<T>::~CudaVector()
{
    this->free();
}

template <typename T> inline
void CudaVector<T>::copy_from_host(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyHostToDevice) );
}

template <typename T> inline
void CudaVector<T>::copy_to_host(T* dst) const {
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
                           reinterpret_cast<const void*>(data_),
                           sizeof(T)*this->size(),
                           cudaMemcpyDeviceToHost) );
}

template <typename T> inline
void CudaVector<T>::copy_from_cuda(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToDevice) );
}

template <typename T> inline
void CudaVector<T>::copy_to_cuda(T* dst) const {
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
                           reinterpret_cast<const void*>(data_),
                           sizeof(T)*this->size(),
                           cudaMemcpyDeviceToDevice) );
}

template <typename T> inline
void CudaVector<T>::copy_from_device(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToDevice) );
}

template <typename T> inline
CudaVector<T>& CudaVector<T>::operator=(const CudaVector<T>& other)
{
    this->copy_from_cuda(other.size(), other.data());
    return *this;
}

template <typename T> inline
CudaVector<T>& CudaVector<T>::operator=(const HostVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
CudaVector<T>& CudaVector<T>::operator=(const PinnedVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
CudaVector<T>& CudaVector<T>::operator=(const std::vector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
void CudaVector<T>::allocate(std::size_t size)
{
    this->free();
    CUDA_CHECK( cudaMalloc(&data_, sizeof(T)*size) );
    capacity_ = size;
}

template <typename T> inline
void CudaVector<T>::free()
{
    if(data_)
        CUDA_CHECK( cudaFree(data_) );
    data_     = nullptr;
    capacity_ = 0;
    size_     = 0;
}

template <typename T> inline
void CudaVector<T>::resize(std::size_t size)
{
    if(capacity_ < size)
        this->allocate(size);
    size_ = size;
}

template <typename T> inline
GlobalRef<T> CudaVector<T>::operator[](std::size_t idx)
{
    return GlobalRef<T>(data_ + idx);
}

template <typename T> inline
ConstGlobalRef<T> CudaVector<T>::operator[](std::size_t idx) const
{
    return ConstGlobalRef<T>(data_ + idx);
}

template <typename T> inline
GlobalRef<T> CudaVector<T>::front()
{
    return GlobalRef<T>(data_);
}

template <typename T> inline
GlobalRef<T> CudaVector<T>::back()
{
    return GlobalRef<T>(data_ + size_ - 1);
}

template <typename T> inline
ConstGlobalRef<T> CudaVector<T>::front() const
{
    return ConstGlobalRef<T>(data_);
}

template <typename T> inline
ConstGlobalRef<T> CudaVector<T>::back() const
{
    return ConstGlobalRef<T>(data_ + size_ - 1);
}


}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::CudaVector<T>& v)
{
    os << rtac::HostVector<T>(v);
    return os;
}

#endif //_DEF_RTAC_BASE_CUDA_DEVICE_VECTOR_H_
