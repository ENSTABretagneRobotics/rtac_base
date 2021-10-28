#ifndef _DEF_RTAC_BASE_TYPES_BOUNDS_H_
#define _DEF_RTAC_BASE_TYPES_BOUNDS_H_

#include <iostream>
#include <array>

namespace rtac { namespace types {

template <typename T>
struct Interval {
    T min;
    T max;

    T length() const {
        return this->max - this->min;
    }
};

template <typename T, std::size_t N>
using Bounds = std::array<Interval<T>, N>;

}; //namespace types
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Interval<T>& interval)
{
    os << "(min : " << interval.min << ", max : " << interval.max << ")";
    return os;
}

template <typename T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const rtac::types::Bounds<T,N>& bounds)
{
    for(auto& i : bounds) {
        os << i << std::endl;
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_BOUNDS_H_
