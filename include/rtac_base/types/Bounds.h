#ifndef _DEF_RTAC_BASE_TYPES_BOUNDS_H_
#define _DEF_RTAC_BASE_TYPES_BOUNDS_H_

#include <iostream>
#include <limits>
#include <type_traits>

#include <rtac_base/cuda_defines.h>

namespace rtac {

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

    Bounds() = default;
    Bounds<T,1>& operator=(const Bounds<T,1>&) = default;

    Bounds(T l, T u) : lower(l), upper(u) {}

    bool operator==(const Bounds<T,1>& other) {
        return this->lower == other.lower && this->upper == other.upper;
    }

    RTAC_HOSTDEVICE T length() const {
        return upper - lower;
    }
    RTAC_HOSTDEVICE T center() const {
        return 0.5*(upper + lower);
    }

    [[deprecated]]
    RTAC_HOSTDEVICE bool is_inside(T value) const { // change to contains
        return lower < value && value < upper;
    }
    RTAC_HOSTDEVICE bool contains(T value) const {
        return lower < value && value < upper;
    }
    RTAC_HOSTDEVICE bool contains(const Bounds<T,1>& other) const {
        return this->lower <= other.lower && other.upper <= this->upper;
    }
    RTAC_HOSTDEVICE Bounds<T,1>& intersect_with(const Bounds<T,1>& other) {
        T m = std::max(this->lower, other.lower);
        T M = std::min(this->upper, other.upper);
        if(m > M) {
            lower = 0; upper = 0;
        }
        else {
            lower = m; upper = M;
        }
        return *this;
    }

    RTAC_HOSTDEVICE void update(T value) {
        lower = std::min(lower, value);
        upper = std::max(upper, value);
    }
    RTAC_HOSTDEVICE void init(T value) {
        lower = value;
        upper = value;
    }
    RTAC_HOSTDEVICE void update(const Bounds<T>& other) {
        lower = std::min(lower, other.lower);
        upper = std::max(upper, other.upper);
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
//    static_assert(false, "rtac::Bounds : invalid number of dimentions");
//};

}; //namespace rtac

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const rtac::Bounds<T>& interval)
{
    os << '[' << interval.lower << ", " << interval.upper << ']'
       << ", L : " << interval.length();
    return os;
}

template <typename T, uint32_t N>
inline std::ostream& operator<<(std::ostream& os, const rtac::Bounds<T,N>& bounds)
{
    for(int i = 0; i < bounds.size(); i++) {
        os << "- " << i << " : " << bounds[i] << std::endl;
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_BOUNDS_H_
