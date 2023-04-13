#ifndef _DEF_RTAC_BASE_CUDA_PINNED_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_PINNED_VECTOR_H_

#include <iostream>
#include <vector>

#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace display {
    template <typename T> class GLVector;
}}

namespace rtac { namespace cuda {

template <typename T>
class CudaVector;

template <typename T>
class PinnedVector
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

    static PinnedVector<T> linspace(T first, T last, std::size_t size) {
        return HostVector<T>::linspace(first, last, size);
    }

    protected:

    T*          data_;
    std::size_t size_;
    std::size_t capacity_;

    void allocate(std::size_t size);
    void free();

    public:

    PinnedVector();
    PinnedVector(std::size_t size);
    PinnedVector(const PinnedVector<T>& other);
    PinnedVector(const HostVector<T>& other);
    PinnedVector(const CudaVector<T>& other);
    PinnedVector(const std::vector<T>& other);
    ~PinnedVector();

    void copy_from_host(std::size_t size, const T* data);
    void copy_to_host(T* dst) const;
    void copy_from_cuda(std::size_t size, const T* data);
    void copy_to_cuda(T* dst) const;

    [[deprecated]]
    void copy_from_device(std::size_t size, const T* data);
    
    PinnedVector& operator=(const PinnedVector<T>& other);
    PinnedVector& operator=(const HostVector<T>& other);
    PinnedVector& operator=(const CudaVector<T>& other);
    PinnedVector& operator=(const std::vector<T>& other);

    void resize(std::size_t size);
    void clear() { this->free(); }

    std::size_t size()     const { return size_;     }
    std::size_t capacity() const { return capacity_; }

    const T* data() const { return data_; }
          T* data()       { return data_; }

    const T* cbegin() const { return data_; }
    const T* begin()  const { return data_; }
          T* begin()        { return data_; }
    const T* cend()   const { return data_ + size_; }
    const T* end()    const { return data_ + size_; }
          T* end()          { return data_ + size_; }

    auto const_view() const { return this->view();                                    }
    auto view()       const { return VectorView<const T>(this->size(), this->data()); }
    auto view()             { return VectorView<T>(this->size(), this->data());       }

    const T& operator[](std::size_t idx) const { return data_[idx]; }
          T& operator[](std::size_t idx)       { return data_[idx]; }

    const T& front() const { return data_[0]; }
          T& front()       { return data_[0]; }
    const T& back()  const { return data_[size_ - 1]; }
          T& back()        { return data_[size_ - 1]; }

    PinnedVector(const display::GLVector<T>& other) { *this = other; }
    PinnedVector& operator=(const display::GLVector<T>& other) {
        this->resize(other.size());
        other.copy_to_host(this->data());
        return *this;
    }
};

// implementation
template <typename T> inline
PinnedVector<T>::PinnedVector() :
    data_(NULL),
    size_(0),
    capacity_(0)
{}

template <typename T> inline
PinnedVector<T>::PinnedVector(std::size_t size) :
    PinnedVector()
{
    this->resize(size);
}

template <typename T> inline
PinnedVector<T>::PinnedVector(const PinnedVector<T>& other) :
    PinnedVector(other.size())
{
    *this = other;
}

template <typename T> inline
PinnedVector<T>::PinnedVector(const HostVector<T>& other) :
    PinnedVector(other.size())
{
    *this = other;
}

template <typename T> inline
PinnedVector<T>::PinnedVector(const CudaVector<T>& other) :
    PinnedVector(other.size())
{
    *this = other;
}

template <typename T> inline
PinnedVector<T>::PinnedVector(const std::vector<T>& other) :
    PinnedVector(other.size())
{
    *this = other;
}

template <typename T> inline
PinnedVector<T>::~PinnedVector()
{
    this->free();
}

template <typename T> inline
void PinnedVector<T>::copy_from_host(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyHostToHost) );
}

template <typename T> inline
void PinnedVector<T>::copy_to_host(T* dst) const {
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
                           reinterpret_cast<const void*>(data_),
                           sizeof(T)*this->size(),
                           cudaMemcpyHostToHost) );
}

template <typename T> inline
void PinnedVector<T>::copy_from_cuda(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToHost) );
}

template <typename T> inline
void PinnedVector<T>::copy_to_cuda(T* dst) const {
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(dst),
                           reinterpret_cast<const void*>(data_),
                           sizeof(T)*this->size(),
                           cudaMemcpyHostToDevice) );
}

template <typename T> inline
void PinnedVector<T>::copy_from_device(std::size_t size, const T* data)
{
    this->resize(size);
    CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(data_),
                           reinterpret_cast<const void*>(data),
                           sizeof(T)*size_,
                           cudaMemcpyDeviceToHost) );
}

template <typename T> inline
PinnedVector<T>& PinnedVector<T>::operator=(const PinnedVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
PinnedVector<T>& PinnedVector<T>::operator=(const HostVector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
PinnedVector<T>& PinnedVector<T>::operator=(const CudaVector<T>& other)
{
    this->copy_from_cuda(other.size(), other.data());
    return *this;
}

template <typename T> inline
PinnedVector<T>& PinnedVector<T>::operator=(const std::vector<T>& other)
{
    this->copy_from_host(other.size(), other.data());
    return *this;
}

template <typename T> inline
void PinnedVector<T>::allocate(std::size_t size)
{
    this->free();
    CUDA_CHECK( cudaMallocHost(&data_, sizeof(T)*size) );
    capacity_ = size;
}

template <typename T> inline
void PinnedVector<T>::free()
{
    CUDA_CHECK( cudaFreeHost(data_) );
    capacity_ = 0;
    size_     = 0;
}

template <typename T> inline
void PinnedVector<T>::resize(std::size_t size)
{
    if(capacity_ < size)
        this->allocate(size);
    size_ = size;
}

}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::PinnedVector<T>& v)
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

#endif //_DEF_RTAC_BASE_CUDA_PINNED_VECTOR_H_
