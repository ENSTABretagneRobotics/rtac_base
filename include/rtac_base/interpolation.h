#ifndef _DEF_RTAC_BASE_INTERPOLATION_H_
#define _DEF_RTAC_BASE_INTERPOLATION_H_

#include <rtac_base/interpolation_impl.h>
#include <rtac_base/types/VectorView.h>

namespace rtac { namespace algorithm {

template <typename T>
class Interpolator
{
    public:
    
    using value_type = T;
    using Vector     = typename InterpolatorInterface<T>::Vector;

    enum Type {
        Nearest,
        Linear,
        CubicSpline
    };

    template <class VectorT>
    static rtac::types::VectorView<T> make_view(VectorT& v);
    template <class VectorT>
    static rtac::types::VectorView<const T> make_view(const VectorT& v);

    protected:

    typename InterpolatorInterface<T>::ConstPtr interpolator_;

    public:

    Interpolator(rtac::types::VectorView<const T> x0,
                 rtac::types::VectorView<const T> y0,
                 Type type = Nearest);

    const Vector& x0() const { return interpolator_->x0(); }
    const Vector& y0() const { return interpolator_->y0(); }
    
    unsigned int size() const { return interpolator_->size(); }
    
    void interpolate(rtac::types::VectorView<const T> x,
                     rtac::types::VectorView<T> y) const;

    template <typename VectorT>
    static Interpolator<T> CreateNearest(const VectorT& x0, const VectorT& y0);
    template <typename VectorT>
    static Interpolator<T> CreateLinear(const VectorT& x0, const VectorT& y0);
    template <typename VectorT>
    static Interpolator<T> CreateCubicSpline(const VectorT& x0, const VectorT& y0);

    template <typename VectorT>
    void interpolate(const VectorT& x, VectorT& y) const;
    template <typename VectorT>
    VectorT interpolate(const VectorT& x) const;
    template <typename VectorT>
    VectorT operator()(const VectorT& x) const;
};


template <typename T> template <class VectorT>
rtac::types::VectorView<T> Interpolator<T>::make_view(VectorT& v)
{
    return rtac::types::VectorView<T>(v.size(), v.data());
}

template <typename T> template <class VectorT>
rtac::types::VectorView<const T> Interpolator<T>::make_view(const VectorT& v)
{
    return rtac::types::VectorView<const T>(v.size(), v.data());
}

template <typename T> 
Interpolator<T>::Interpolator(rtac::types::VectorView<const T> x0,
                              rtac::types::VectorView<const T> y0,
                              Type type) :
    interpolator_(nullptr)
{
    switch(type) {
        default:
            throw std::runtime_error("rtac::algorithm::Interpolator : invalid interpolation type");
            break;
        case Nearest:
            interpolator_ = std::make_shared<InterpolatorNearest<T>>(x0, y0);
            break;
        case Linear:
            interpolator_ = std::make_shared<InterpolatorLinear<T>>(x0, y0);
            break;
        case CubicSpline:
            interpolator_ = std::make_shared<InterpolatorCubicSpline<T>>(x0, y0);
            break;
    }
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateNearest(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(make_view(x0), make_view(y0), Nearest);
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateLinear(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(make_view(x0), make_view(y0), Linear);
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateCubicSpline(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(make_view(x0), make_view(y0), CubicSpline);
}
    
template <typename T> 
void Interpolator<T>::interpolate(rtac::types::VectorView<const T> x,
                                  rtac::types::VectorView<T> y) const
{
    interpolator_->interpolate(x, y);
}

template <typename T> template <typename VectorT>
void Interpolator<T>::interpolate(const VectorT& x, VectorT& y) const
{
    this->interpolate(make_view(x), make_view(y));
}

template <typename T> template <typename VectorT>
VectorT Interpolator<T>::interpolate(const VectorT& x) const
{
    VectorT y(x.size());
    this->interpolate(x,y);
    return y;
}

template <typename T> template <typename VectorT>
VectorT Interpolator<T>::operator()(const VectorT& x) const
{
    return this->interpolate(x);
}

}; //namespace algorithm
}; //namespace rtac

#endif //_DEF_RTAC_BASE_INTERPOLATION_H_
