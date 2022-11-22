#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <type_traits>
#include <rtac_base/cuda_defines.h>

namespace rtac {

template <typename T>
class VectorView
{
    public:
    
    using value_type = T;

    protected:
    
    T*          data_;
    std::size_t size_;

    public:

    VectorView(std::size_t size = 0, T* data = nullptr) : data_(data), size_(size) {}
    template <template<typename>class VectorT>
    VectorView(VectorT<T>& vector) : VectorView(vector.size(), vector.data()) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }

    RTAC_HOSTDEVICE value_type* data()  { return data_; }
    RTAC_HOSTDEVICE value_type* begin() { return data_; }
    RTAC_HOSTDEVICE value_type* end()   { return data_ + size_; }
    
    RTAC_HOSTDEVICE const T& operator[](std::size_t idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE const T& front()                     const { return data_[0]; }
    RTAC_HOSTDEVICE const T& back()                      const { return data_[size_ - 1]; }
    
    // modifiers
    RTAC_HOSTDEVICE T& operator[](std::size_t idx) { return data_[idx]; }
    RTAC_HOSTDEVICE T& front()                     { return data_[0]; }
    RTAC_HOSTDEVICE T& back()                      { return data_[size_ - 1]; }
};

/**
 * Specialization of VectorView to handle const Vectors. Modifiers are removed.
 */
template <typename T>
class VectorView<const T>
{
    public:
    
    using value_type = T;

    protected:
    
    const T*    data_;
    std::size_t size_;

    public:

    VectorView(std::size_t size = 0, const T* data = nullptr) : data_(data), size_(size) {}
    template <template<typename>class VectorT>
    VectorView(const VectorT<T>& vector) : VectorView(vector.size(), vector.data()) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }

    RTAC_HOSTDEVICE const T& operator[](std::size_t idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE const T& front()                     const { return data_[0]; }
    RTAC_HOSTDEVICE const T& back()                      const { return data_[size_ - 1]; }
};

// have to define these proxy types. nvcc does not play well with templates...
template <class VectorT>
auto make_view(VectorT& vector)
{
    return VectorView<typename VectorT::value_type>(vector.size(), vector.data());
}

template <class VectorT>
auto make_view(const VectorT& vector)
{
    return VectorView<const typename VectorT::value_type>(vector.size(), vector.data());
}

}; //namespace rtac

#endif  //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_


