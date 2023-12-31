#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <type_traits>
#include <rtac_base/cuda_defines.h>

namespace rtac {

/**
 * Small container to store a pointer to a continuous memory location an its
 * size, along with a regular array interface (size getter, iterator,
 * subscripting...)
 *
 * Primary purpose is to pass data around in a CUDA kernel.
 *
 * Caution : this type is only valid as long as the original memory location is
 * valid. There are no memory protections.
 */
template <typename T>
class VectorView
{
    public:
    
    using value_type = T;

    protected:
    
    T*          data_;
    std::size_t size_;

    public:

    VectorView() = default;
    VectorView<T>& operator=(const VectorView<T>&) = default;

    RTAC_HOSTDEVICE VectorView(std::size_t size, T* data) : data_(data), size_(size) {}
    template <template<typename>class VectorT> RTAC_HOSTDEVICE
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

    VectorView() = default;
    VectorView<const T>& operator=(const VectorView<const T>&) = default;

    VectorView(const VectorView<T>& other) : size_(other.size()), data_(other.data()) {}
    VectorView<const T>& operator=(const VectorView<T>& other) {
        size_ = other.size();
        data_ = other.data();
        return *this;
    }

    RTAC_HOSTDEVICE VectorView(std::size_t size, const T* data) : data_(data), size_(size) {}
    
    template <template<typename>class VectorT> RTAC_HOSTDEVICE // [[deprecated]]
    VectorView(const VectorT<T>& vector) : VectorView(vector.size(), vector.data()) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }

    RTAC_HOSTDEVICE const T& operator[](std::size_t idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE const T& front()                     const { return data_[0]; }
    RTAC_HOSTDEVICE const T& back()                      const { return data_[size_ - 1]; }
};

/**
 * This works as an alias for VectorView<const T>
 *
 * Mostly used when argument deduction fails.
 */
template <typename T>
struct ConstVectorView : public VectorView<const T>
{
    using value_type = T;
    using VectorView<const T>::VectorView;
};

/**
 * Makes a VectorView with automatic template argument deduction for c++14.
 * Automatic template argument deduction for constructors is a c++17 feature.
 */
template <typename T> inline RTAC_HOSTDEVICE
VectorView<T> make_vector_view(unsigned int size, T* data) {
    return VectorView<T>(size, data);
}

/**
 * Makes a VectorView with automatic template argument deduction for c++14.
 * Automatic template argument deduction for constructors is a c++17 feature.
 */
template <typename T> inline RTAC_HOSTDEVICE
VectorView<const T> make_vector_view(unsigned int size, const T* data) { 
    return VectorView<const T>(size, data);
}

/**
 * Makes a VectorView with automatic template argument deduction for c++14.
 * Automatic template argument deduction for constructors is a c++17 feature.
 */
template <typename T> inline RTAC_HOSTDEVICE
VectorView<const T> make_const_vector_view(unsigned int size, T* data) { 
    return VectorView<const T>(size, data);
}

/**
 * Makes a VectorView with automatic template argument deduction for c++14.
 * Automatic template argument deduction for constructors is a c++17 feature.
 */
template <typename T> inline RTAC_HOSTDEVICE
VectorView<const T> make_const_vector_view(unsigned int size, const T* data) { 
    return VectorView<const T>(size, data);
}

}; //namespace rtac

#endif  //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_


