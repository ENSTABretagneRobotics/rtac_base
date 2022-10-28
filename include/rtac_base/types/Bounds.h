#ifndef _DEF_RTAC_BASE_TYPES_BOUNDS_H_
#define _DEF_RTAC_BASE_TYPES_BOUNDS_H_

#include <iostream>
#include <limits>
#include <type_traits>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

template <typename T, uint32_t SizeV = 1>
struct Bounds
{
    static constexpr uint32_t Size = SizeV;

    Bounds<T> intervals_[Size];

    RTAC_HOSTDEVICE uint32_t size() const { return Size; }

    RTAC_HOSTDEVICE Bounds<T>& operator[](uint32_t idx) {
        return intervals_[idx];
    }
    RTAC_HOSTDEVICE const Bounds<T>& operator[](uint32_t idx) const {
        return intervals_[idx];
    }

    RTAC_HOSTDEVICE Bounds<T>*       begin()       { return intervals_; }
    RTAC_HOSTDEVICE Bounds<T>*       end()         { return this->begin() + Size; }
    RTAC_HOSTDEVICE const Bounds<T>* begin() const { return intervals_; }
    RTAC_HOSTDEVICE const Bounds<T>* end()   const { return this->begin() + Size; }

    RTAC_HOSTDEVICE static Bounds<T,Size> oo() {
        Bounds<T,Size> res;
        for(int i = 0; i < Size; i++) {
            res[i] = Bounds<T>::oo();
        }
        return res;
    }

    RTAC_HOSTDEVICE static Bounds<T,Size> Zero() {
        Bounds<T,Size> res;
        for(int i = 0; i < Size; i++) {
            res[i] = Bounds<T>::Zero();
        }
        return res;
    }
};

template <typename T>
struct Bounds<T,1>
{
    T lower;
    T upper;

    RTAC_HOSTDEVICE T length() const {
        return upper - lower;
    }

    RTAC_HOSTDEVICE bool is_inside(T value) const {
        return lower < value && value < upper;
    }

    RTAC_HOSTDEVICE void update(T value) {
        lower = min(lower, value);
        upper = max(upper, value);
    }
    RTAC_HOSTDEVICE void init(T value) {
        lower = value;
        upper = value;
    }
    RTAC_HOSTDEVICE void update(const Bounds<T>& other) {
        lower = min(lower, other.lower);
        upper = max(upper, other.upper);
    }

    RTAC_HOSTDEVICE static Bounds<T> oo() {
        if(std::is_floating_point<T>::value) {
            return Bounds<T>{-std::numeric_limits<T>::infinity(),
                                std::numeric_limits<T>::infinity()};
        }
        else if(std::is_integral<T>::value) {
            return Bounds<T>{std::numeric_limits<T>::min(),
                               std::numeric_limits<T>::max()};
        }
        else {
            throw std::runtime_error("No oo defined for this type");
        }
    }

    RTAC_HOSTDEVICE static Bounds<T> Zero() {
        return Bounds<T>{0,0};
    }
};

//template <typename T>
//struct Bounds<T,0> {
//    static_assert(false, "rtac::types::Bounds : invalid number of dimentions");
//};

}; //namespace types
}; //namespace rtac

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const rtac::types::Bounds<T>& interval)
{
    os << '[' << interval.lower << ", " << interval.upper << ']'
       << ", L : " << interval.length();
    return os;
}

template <typename T, uint32_t N>
inline std::ostream& operator<<(std::ostream& os, const rtac::types::Bounds<T,N>& bounds)
{
    for(int i = 0; i < bounds.size(); i++) {
        os << "- " << i << " : " << bounds[i] << std::endl;
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_BOUNDS_H_
