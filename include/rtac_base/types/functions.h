#ifndef _DEF_RTAC_BASE_CONTAINERS_CONTINUOUS_IMAGE_H_
#define _DEF_RTAC_BASE_CONTAINERS_CONTINUOUS_IMAGE_H_

#include <type_traits>
#include <cmath>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>

namespace rtac {

template <class Derived>
struct Function1D
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return reinterpret_cast<const Derived*>(this);
    }

    RTAC_HOSTDEVICE bool is_in_domain(float x) const {
        return this->domain().is_inside(x);
    }

    // these 2 methods must be reimplemented in a subclass
    RTAC_HOSTDEVICE Bounds<float> domain() const {
        return this->cast()->x_domain();
    }
    RTAC_HOSTDEVICE auto operator()(float x) const {
        return this->cast()->operator()(x);
    }
};

struct LinearFunction1D : public Function1D<LinearFunction1D>
{
    float a_, b_;
    Bounds<float> domain_;

    RTAC_HOSTDEVICE Bounds<float> domain() const { return domain_; }
    RTAC_HOSTDEVICE float operator()(float x) const { return fmaf(a_, x, b_); }

    RTAC_HOSTDEVICE static LinearFunction1D make(float a, float b,
                                                 const Bounds<float>& domain) {
        //return LinearFunction1D{{}, a, b, domain};
        // Have to do this because Jetson TX2 make cuda version is 10.2, so max c++ std is 14
        LinearFunction1D res;
        res.a_ = a; res.b_ = b; res.domain_ = domain;
        return res;
    }
};

template <class Derived>
struct Function2D
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return reinterpret_cast<const Derived*>(this);
    }

    RTAC_HOSTDEVICE bool is_in_domain(float x, float y) const {
        return this->x_domain().is_inside(x) && this->y_domain().is_inside(y);
    }

    // these 3 method must be reimplemented in a subclass
    RTAC_HOSTDEVICE Bounds<float> x_domain() const { 
        return this->cast()->x_domain();
    }
    RTAC_HOSTDEVICE Bounds<float> y_domain() const { 
        return this->cast()->y_domain();
    }
    RTAC_HOSTDEVICE auto operator()(float x, float y) const {
        return this->cast()->operator()(x ,y);
    }
};

template <class Derived>
struct IsFunction1D {
    static constexpr bool value = std::is_base_of<
        Function1D<Derived>, Derived>::value;
};

template <class Derived>
struct IsFunction2D {
    static constexpr bool value = std::is_base_of<
        Function2D<Derived>, Derived>::value;
};

} //namespace rtac

#endif //_DEF_RTAC_BASE_CONTAINERS_CONTINUOUS_IMAGE_H_
