#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <type_traits>
#include <rtac_base/type_utils.h>
#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

namespace details {

// /**
//  * These implements operator[] in a class only if VectorT also implements it.
//  * 
//  * Useful for DeviceVector where operator[] is defined only on device side.
//  */
// template <class Derived, typename T, bool IsSubscriptable>
// struct subscript_traits {};

template <class Derived>
struct const_subscriptable_traits
{
    private:
    // this is for convenience.
    RTAC_HOSTDEVICE const Derived* derived() const {
        return reinterpret_cast<const Derived*>(this);
    }

    public:

    RTAC_HOSTDEVICE const auto& operator[](std::size_t idx) const {
        return this->derived()->data()[idx];
    }

    RTAC_HOSTDEVICE const auto& front() const { return (*this)[0]; }
    RTAC_HOSTDEVICE const auto& back() const  { return (*this)[this->derived()->size() - 1]; }
};

template <class Derived>
struct mutable_subscriptable_traits
{
    private:
    // this is for convenience.
    RTAC_HOSTDEVICE Derived* derived() {
        return reinterpret_cast<Derived*>(this);
    }
    RTAC_HOSTDEVICE const Derived* derived() const {
        return reinterpret_cast<const Derived*>(this);
    }

    public:

    RTAC_HOSTDEVICE auto& operator[](std::size_t idx) { 
        return this->derived()->data()[idx];
    }
    RTAC_HOSTDEVICE auto& front() { return (*this)[0]; }
    RTAC_HOSTDEVICE auto& back()  { return (*this)[this->derived()->size() - 1]; }

    RTAC_HOSTDEVICE const auto& operator[](std::size_t idx) const {
        return this->derived()->data()[idx];
    }
    RTAC_HOSTDEVICE const auto& front() const { return (*this)[0]; }
    RTAC_HOSTDEVICE const auto& back() const  { return (*this)[this->derived()->size() - 1]; }
};


};

template <class Derived, typename T, bool Subscriptable>
struct subscriptable_traits {};

template <class Derived, typename T>
struct subscriptable_traits<Derived, T, true> :
    public details::mutable_subscriptable_traits<Derived>
{};

template <class Derived, typename T>
struct subscriptable_traits<Derived, const T, true> :
    public details::const_subscriptable_traits<Derived>
{};

/**
 * Generic VectorView type. Mutable by default. Immutable is implemented as a
 * specialization and removes modifiers.
 *
 * The subscript_traits add the operator[] only if it is defined for VectorT.
 */
template <typename T, bool Subscriptable = true>
class VectorView : public subscriptable_traits<VectorView<T>, T, Subscriptable>
{
    public:
    
    using value_type = T;

    protected:
    
    T*          data_;
    std::size_t size_;

    public:

    VectorView(std::size_t size = 0, T* data = nullptr) : data_(data), size_(size) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }

    RTAC_HOSTDEVICE value_type* data()  { return data_; }
    RTAC_HOSTDEVICE value_type* begin() { return data_; }
    RTAC_HOSTDEVICE value_type* end()   { return data_ + size_; }
};

/**
 * Specialization of VectorView to handle const Vectors. Modifiers are removed.
 */
template <typename T, bool Subscriptable>
class VectorView<const T, Subscriptable>
    : public subscriptable_traits<VectorView<const T>, const T, Subscriptable>
{
    public:
    
    using value_type = T;

    protected:
    
    const T*    data_;
    std::size_t size_;

    public:

    VectorView(std::size_t size = 0, const T* data = nullptr) : data_(data), size_(size) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }
};

template <typename VectorT>
auto make_view(const VectorT& vector)
{
    return VectorView<const typename VectorT::value_type,
                      is_subscriptable<VectorT>::value>(vector.size(), vector.data());
}

template <typename VectorT>
auto make_view(VectorT& vector)
{
    return VectorView<typename VectorT::value_type,
                      is_subscriptable<VectorT>::value>(vector.size(), vector.data());
}

}; //namespace types
}; //namespace rtac

#endif  //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_


