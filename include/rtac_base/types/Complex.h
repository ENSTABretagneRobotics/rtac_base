#ifndef _DEF_RTAC_BASE_TYPES_COMPLEX_H_
#define _DEF_RTAC_BASE_TYPES_COMPLEX_H_

#include <iostream>

#include <rtac_base/cuda_defines.h>

namespace rtac {

/**
 * Have to make a Complex class because cuda does not implement functions for
 * std::complex class and thrust::complex (and thrust in general) is causing
 * linking issues when in use with the OptiX framework (multiple definition of
 * the __global__ EmptyKernel function, declared and defined in the
 * cuda/include/cub/util_device.cuh
 *
 * The interface of this Complex<T> type is a subset of std::complex.
 * Non-complex number are treated as complex number with imaginary part equal to 0
 */
template <typename T>
class Complex
{
    public:

    using value_type = T;

    protected:

    T real_;
    T imag_;

    public:

    RTAC_HOSTDEVICE constexpr Complex(const T& re = T(), const T& im = T()) : real_(re), imag_(im) {}
    RTAC_HOSTDEVICE constexpr Complex(const Complex<T>& other) : real_(other.real_), imag_(other.imag_) {}
    template <class T2>
    RTAC_HOSTDEVICE constexpr Complex(const Complex<T2>& other) : real_(other.real_), imag_(other.imag_) {}

    RTAC_HOSTDEVICE Complex<T>& operator=(const T& x)              { real_ = x; imag_ = 0; return *this; }
    RTAC_HOSTDEVICE Complex<T>& operator=(const Complex<T>& other) { real_ = other.real_; imag_ = other.imag_; return *this; }
    template <class T2>
    Complex<T>& operator=(const Complex<T2>& other) { real_ = other.real_; imag_ = other.imag_; return *this; }

    RTAC_HOSTDEVICE constexpr T real()  const { return real_; }
    RTAC_HOSTDEVICE void        real(T value) { real_ = value; }
    RTAC_HOSTDEVICE constexpr T imag()  const { return imag_; }
    RTAC_HOSTDEVICE void        imag(T value) { imag_ = value; }

    RTAC_HOSTDEVICE Complex<T>& operator+=(const T& other) { real_ += other; return *this; }
    RTAC_HOSTDEVICE Complex<T>& operator-=(const T& other) { real_ -= other; return *this; }
    RTAC_HOSTDEVICE Complex<T>& operator*=(const T& other) { real_ *= other; imag_ *= other; return *this; }
    RTAC_HOSTDEVICE Complex<T>& operator/=(const T& other) { real_ /= other; imag_ /= other; return *this; }

    template <class T2> RTAC_HOSTDEVICE
    Complex<T>& operator+=(const Complex<T2>& other) { real_ += other.real_; imag_ += other.imag_; return *this; }
    template <class T2> RTAC_HOSTDEVICE
    Complex<T>& operator-=(const Complex<T2>& other) { real_ -= other.real_; imag_ -= other.imag_; return *this; }
    template <class T2> RTAC_HOSTDEVICE
    Complex<T>& operator*=(const Complex<T2>& other) { 
        *this = Complex<T>(real_*other.real_ - imag_*other.imag_, real_*other.imag_ + imag_*other.real_); 
        return *this;
    }
    template <class T2> RTAC_HOSTDEVICE
    Complex<T>& operator/=(const Complex<T2>& other) {
        T2 n = other.real_*other.real_ + other.imag_*other.imag_;
        *this = Complex<T>((real_*other.real_ + imag_*other.imag_) / n,
                           (imag_*other.real_ - real_*other.imag_) / n);
        return *this;
    }
};


} //namespace rtac

template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator+(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs) += rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator+(const rtac::Complex<T>& lhs, const T& rhs)                { return rtac::Complex<T>(lhs) += rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator+(const T& lhs,                const rtac::Complex<T>& rhs) { return rtac::Complex<T>(rhs) += lhs; }

template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator-(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs)    -= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator-(const rtac::Complex<T>& lhs, const T& rhs)                { return rtac::Complex<T>(lhs)    -= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator-(const T& lhs,                const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs, 0) -= rhs; }

template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator*(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs) *= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator*(const rtac::Complex<T>& lhs, const T& rhs)                { return rtac::Complex<T>(lhs) *= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator*(const T& lhs,                const rtac::Complex<T>& rhs) { return rtac::Complex<T>(rhs) *= lhs; }

template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator/(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs)    /= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator/(const rtac::Complex<T>& lhs, const T& rhs)                { return rtac::Complex<T>(lhs)    /= rhs; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator/(const T& lhs,                const rtac::Complex<T>& rhs) { return rtac::Complex<T>(lhs, 0) /= rhs; }

template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator+(const rtac::Complex<T>& value) { return value; }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> operator-(const rtac::Complex<T>& value) { return rtac::Complex<T>(-value.real(), -value.imag()); }

template <class T> RTAC_HOSTDEVICE inline bool operator==(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return lhs.real() == rhs.real() && lhs.imag() == lhs.imag(); }
template <class T> RTAC_HOSTDEVICE inline bool operator==(const rtac::Complex<T>& lhs, const T& rhs)                { return lhs.real() == rhs        && lhs.imag() == 0; }
template <class T> RTAC_HOSTDEVICE inline bool operator==(const T& lhs,                const rtac::Complex<T>& rhs) { return rhs.real() == lhs        && rhs.imag() == 0; }

template <class T> RTAC_HOSTDEVICE inline bool operator!=(const rtac::Complex<T>& lhs, const rtac::Complex<T>& rhs) { return !(lhs == rhs); }
template <class T> RTAC_HOSTDEVICE inline bool operator!=(const rtac::Complex<T>& lhs, const T& rhs)                { return !(lhs == rhs); }
template <class T> RTAC_HOSTDEVICE inline bool operator!=(const T& lhs,                const rtac::Complex<T>& rhs) { return !(lhs == rhs); }

template <class T> RTAC_HOSTDEVICE inline T real(const rtac::Complex<T>& value) { return value.real(); }
template <class T> RTAC_HOSTDEVICE inline T imag(const rtac::Complex<T>& value) { return value.imag(); }
template <class T> RTAC_HOSTDEVICE inline T norm(const rtac::Complex<T>& value) { return value.real()*value.real() + value.imag()*value.imag(); }
template <class T> RTAC_HOSTDEVICE inline T abs (const rtac::Complex<T>& value) { return sqrt(norm(value)); }
template <class T> RTAC_HOSTDEVICE inline T arg (const rtac::Complex<T>& value) { return atan2(value.imag(), value.real()); }
template <class T> RTAC_HOSTDEVICE inline rtac::Complex<T> conj(const rtac::Complex<T>& value) { return rtac::Complex<T>(value.real(), -value.imag()); }

template <class T> RTAC_HOSTDEVICE inline T polar(const T& r, const T& theta = T()) { return rtac::Complex<T>(r*cos(theta), r*sin(theta)); }

template <class T> inline std::ostream& operator<<(std::ostream& os, const rtac::Complex<T>& value) {
    os << '(' << value.real() << ',' << value.imag() << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_COMPLEX_H_
