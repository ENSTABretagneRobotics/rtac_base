#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <type_traits>
#include <rtac_base/type_utils.h>
#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

namespace details {

/**
 * These implements operator[] in a class only if VectorT also implements it.
 * 
 * Useful for DeviceVector where operator[] is defined only on device side.
 */
template <class Derived, class VectorT, bool IsSubscriptable>
struct subscript_traits {};

template <class Derived, class VectorT>
struct subscript_traits<Derived, VectorT, true>
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
    RTAC_HOSTDEVICE const auto& operator[](std::size_t idx) const {
        return this->derived()->data()[idx];
    }

    RTAC_HOSTDEVICE auto&       front()       { return (*this)[0]; }
    RTAC_HOSTDEVICE const auto& front() const { return (*this)[0]; }
    RTAC_HOSTDEVICE auto& back()              { return (*this)[this->derived()->size() - 1]; }
    RTAC_HOSTDEVICE const auto& back()  const { return (*this)[this->derived()->size() - 1]; }
};


template <class Derived, class VectorT>
struct subscript_traits<Derived, const VectorT, true>
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

};

template <class Derived, class VectorT>
struct subscript_traits {
    using type = details::subscript_traits<Derived, VectorT, is_subscriptable<VectorT>::value>;
};

/**
 * Generic VectorView type. Mutable by default. Immutable is implemented as a
 * specialization and removes modifiers.
 *
 * The subscript_traits add the operator[] only if it is defined for VectorT.
 */
template <typename VectorT>
class VectorView : public subscript_traits<VectorView<VectorT>, VectorT>::type
{
    public:
    
    using value_type = typename VectorT::value_type;

    protected:
    
    value_type* data_;
    std::size_t size_;

    public:

    VectorView(VectorT& vect) : data_(vect.data()), size_(vect.size()) {}

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
template <typename VectorT>
class VectorView<const VectorT>
    : public subscript_traits<VectorView<const VectorT>, const VectorT>::type
{
    public:
    
    using value_type = typename VectorT::value_type;

    protected:
    
    const value_type* data_;
    std::size_t       size_;

    public:

    VectorView(const VectorT& vect) : data_(vect.data()), size_(vect.size()) {}

    RTAC_HOSTDEVICE std::size_t size() const { return size_; }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_; }
    RTAC_HOSTDEVICE const value_type* begin() const { return data_; }
    RTAC_HOSTDEVICE const value_type* end()   const { return data_ + size_; }
};

}; //namespace types
}; //namespace rtac

#endif  //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_


