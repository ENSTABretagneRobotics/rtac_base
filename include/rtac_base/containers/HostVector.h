#ifndef _DEF_RTAC_BASE_HOST_VECTOR_H_
#define _DEF_RTAC_BASE_HOST_VECTOR_H_

#include <cstring>
#include <vector>
#include <iostream>

#include <rtac_base/containers/VectorView.h>

namespace rtac {

// some forward declarations to avoid hard dependencies
namespace cuda {
    template <typename T> class CudaVector;
    template <typename T> class PinnedVector;
} //namespace cuda

namespace display {
    template <typename T> class GLVector;
}

/**
 * Wrapper around a std::vector to unify the interface with other rtac vector
 * types such as VectorView or cuda::CudaVector.
 */
template <typename T>
class HostVector
{
    public:

    using value_type = T;

    static HostVector<T> linspace(T first, T last, std::size_t size) {
        HostVector<T> res(size);
        for(std::size_t i = 0; i < size; i++) {
            res[i] = ((last - first)*i) / (size - 1) + first;
        }
        return res;
    }

    protected:

    std::vector<T> data_;

    public:

    HostVector() {}
    HostVector(std::size_t size)            : data_(size) {}
    HostVector(std::size_t size, T value)   : data_(size, value) {}
    HostVector(const HostVector<T>& other)  : data_(other.data_) {}
    HostVector(const std::vector<T>& other) : data_(other) {}

    HostVector& operator=(const HostVector<T>& other)  { data_ = other.data_; return *this; }
    HostVector& operator=(const std::vector<T>& other) { data_ = other;       return *this; }

    void copy_from_host(std::size_t size, const T* data) {
        this->resize(size);
        std::memcpy(this->data(), data, size*sizeof(T));
    }
    void copy_to_host(T* dst) const {
        std::memcpy(dst, this->data(), this->size()*sizeof(T));
    }
    
    [[deprecated]]
    void copy(std::size_t size, const T* data) { // change to assign ?
        this->resize(size);
        std::memcpy(this->data(), data, size*sizeof(T));
    }
    [[deprecated]]
    void copy_to(T* dst) const {
        std::memcpy(dst, this->data(), this->size()*sizeof(T));
    }

    void resize(std::size_t size) { data_.resize(size); }
    void clear()                  { data_.clear();      }

    std::size_t size()     const { return data_.size();     }
    std::size_t capacity() const { return data_.capacity(); }

    const T* data() const { return data_.data(); }
    T*       data()       { return data_.data(); }

    T*       begin()       { return data_.data();                }
    T*       end()         { return data_.data() + data_.size(); }
    const T* begin() const { return data_.data();                }
    const T* end()   const { return data_.data() + data_.size(); }

    T&       operator[](std::size_t idx)       { return data_[idx]; }
    const T& operator[](std::size_t idx) const { return data_[idx]; }

    T&       front()       { return data_.front(); }
    T&       back()        { return data_.back();  }
    const T& front() const { return data_.front(); }
    const T& back()  const { return data_.back();  }

    auto view() const { return VectorView<const T>(this->size(), this->data()); }
    auto view()       { return VectorView<T>(this->size(), this->data()); }

    HostVector(const cuda::CudaVector<T>& other) { *this = other; }
    HostVector& operator=(const cuda::CudaVector<T>& other) {
        this->resize(other.size());
        other.copy_to_host(this->data());
        return *this;
    }
    HostVector(const cuda::PinnedVector<T>& other) { *this = other; }
    HostVector& operator=(const cuda::PinnedVector<T>& other) {
        this->resize(other.size());
        other.copy_to_host(this->data());
        return *this;
    }
    HostVector(const display::GLVector<T>& other) { *this = other; }
    HostVector& operator=(const display::GLVector<T>& other) {
        this->resize(other.size());
        other.copy_to_host(this->data());
        return *this;
    }
};

} //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::HostVector<T>& v)
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

#endif //_DEF_RTAC_BASE_HOST_VECTOR_H_
