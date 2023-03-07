#ifndef _DEF_RTAC_BASE_TYPES_LINSPACE_H_
#define _DEF_RTAC_BASE_TYPES_LINSPACE_H_

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>

namespace rtac {

/**
 * Simple type to hold information on a linearly sampled interval.
 */
template <typename T>
struct Linspace
{
    Bounds<T> bounds_;
    unsigned int size_;

    Linspace() = default;
    Linspace<T>& operator=(const Linspace<T>&) = default;

    Linspace(const Bounds<T>& bounds, unsigned int size) :
        bounds_(bounds), size_(size)
    {}
    Linspace(T lower, T upper, unsigned int size) :
        Linspace(Bounds<T>(lower, upper), size)
    {}

    RTAC_HOSTDEVICE const Bounds<T>& bounds() const { return bounds_; }
    RTAC_HOSTDEVICE unsigned int     size()   const { return size_; }
    RTAC_HOSTDEVICE T lower() const { return bounds_.lower; }
    RTAC_HOSTDEVICE T upper() const { return bounds_.upper; }

    RTAC_HOSTDEVICE T length()     const { return bounds_.length(); }
    RTAC_HOSTDEVICE T resolution() const { return bounds_.length() / (size_ - 1); }
    RTAC_HOSTDEVICE T operator[](unsigned int idx) const {
        return this->resolution() * idx + bounds_.lower;
    }
};

} //namespace rtac

template <typename T> inline
std::ostream& operator<<(std::ostream& os, const rtac::Linspace<T>& linspace)
{
    os << "Linspace [" << linspace.lower() << ", " << linspace.upper()
       << "], size : " << linspace.size();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_LINSPACE_H_
