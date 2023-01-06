#ifndef _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
#define _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_

#include <cstdint>
#include <cmath>
#include <type_traits>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/containers/Image.h>
#include <rtac_base/containers/DimExpression.h>
#include <rtac_base/containers/utilities.h>

namespace rtac {

template <typename Derived>
struct ScaledImageExpression : public ImageExpression<Derived>
{
    RTAC_HOSTDEVICE const auto& width_dim() const  { return this->cast()->width_dim();  }
    RTAC_HOSTDEVICE const auto& height_dim() const { return this->cast()->height_dim(); }

    RTAC_HOSTDEVICE uint32_t width()  const { return this->width_dim().size();  }
    RTAC_HOSTDEVICE uint32_t height() const { return this->height_dim().size(); }
};

template <typename T, class WDimT, class HDimT>
class ScaledImageView : public ScaledImageExpression< ScaledImageView<T,WDimT,HDimT> >
{
    public:
    
    using value_type = T;
    using WidthDim   = WDimT;
    using HeightDim  = HDimT;
    using Shape      = rtac::Shape<uint32_t>;

    protected:

    T*        data_;
    WidthDim  wDim_;
    HeightDim hDim_; 

    public:

    ScaledImageView(const WidthDim& wDim,
                    const HeightDim& hDim,
                    value_type* data) :
        data_(data), wDim_(wDim), hDim_(hDim)
    {}
    
    RTAC_HOSTDEVICE const T* data() const { return data_;  }
    RTAC_HOSTDEVICE T*       data()       { return data_;  }

    RTAC_HOSTDEVICE uint32_t step() const { return this->width(); }

    RTAC_HOSTDEVICE const WidthDim&  width_dim()  const { return wDim_; }
    RTAC_HOSTDEVICE const HeightDim& height_dim() const { return hDim_; }
};

template <typename T, class WDimT, class HDimT>
class ScaledImageView<const T, WDimT, HDimT>
    : public ScaledImageExpression< ScaledImageView<const T, WDimT, HDimT> >
{
    public:
    
    using value_type = T;
    using WidthDim   = WDimT;
    using HeightDim  = HDimT;
    using Shape      = rtac::Shape<uint32_t>;

    protected:

    const T*  data_;
    WidthDim  wDim_;
    HeightDim hDim_; 

    public:

    ScaledImageView(const WidthDim& wDim,
                    const HeightDim& hDim,
                    const T* data) :
        data_(data), wDim_(wDim), hDim_(hDim)
    {}
    
    RTAC_HOSTDEVICE const T* data() const { return data_;  }

    RTAC_HOSTDEVICE uint32_t step() const { return this->width(); }

    RTAC_HOSTDEVICE const WidthDim&  width_dim()  const { return wDim_; }
    RTAC_HOSTDEVICE const HeightDim& height_dim() const { return hDim_; }
};

template <typename T, class WDimT, class HDimT>
ScaledImageView<T, WDimT, HDimT>
    make_scaled_image_view(const WDimT& wDim, const HDimT& hDim, T* data)
{
    return ScaledImageView<T, WDimT, HDimT>(wDim, hDim, data);
}

template <typename T, class WDimT, class HDimT>
ScaledImageView<const T, WDimT, HDimT>
    make_scaled_image_view(const WDimT& wDim, const HDimT& hDim, const T* data)
{
    return ScaledImageView<const T, WDimT, HDimT>(wDim, hDim, data);
}


template <typename T, class WDimT, class HDimT,
          template<typename>class VectorT>
class ScaledImage : public ScaledImageExpression< ScaledImage<T, WDimT, HDimT, VectorT> >
{
    public:
    
    using value_type = T;
    using WidthDim   = WDimT;
    using HeightDim  = HDimT;
    using Shape      = rtac::Shape<uint32_t>;

    protected:

    VectorT<T> data_;
    WidthDim   wDim_;
    HeightDim  hDim_; 

    public:

    template <template<typename>class VectorT2>
    ScaledImage(const WidthDim& wDim,
                const HeightDim& hDim,
                const VectorT2<T>& data) :
        data_(data), wDim_(wDim), hDim_(hDim)
    {
        if(data_.size() != wDim_.size() * hDim_.size()) {
            throw std::runtime_error("Inconsistent sizes for ScaledImage");
        }
    }

    template <template<typename>class VectorT2>
    ScaledImage<T,WDimT,HDimT,VectorT>&
        operator=(const ScaledImage<T,WDimT,HDimT,VectorT2>& other)
    {
        data_ = other.data_;
        wDim_ = other.wDim_;
        hDim_ = other.hDim_;
        return *this;
    }

    const VectorT<T>& container() const { return data_; }
    VectorT<T>&       container()       { return data_; }
    
    RTAC_HOSTDEVICE const T* data() const { return data_.data();  }
    RTAC_HOSTDEVICE T*       data()       { return data_.data();  }

    RTAC_HOSTDEVICE uint32_t step() const { return this->width(); }

    RTAC_HOSTDEVICE const WidthDim&  width_dim()  const { return wDim_; }
    RTAC_HOSTDEVICE const HeightDim& height_dim() const { return hDim_; }

    RTAC_HOSTDEVICE auto view() const {
        return make_scaled_image_view(make_view(wDim_), make_view(hDim_), this->data());
    }
    RTAC_HOSTDEVICE auto view() {
        return make_scaled_image_view(make_view(wDim_), make_view(hDim_), this->data());
    }
};

template <typename T, class WDimT, class HDimT, template<typename>class V>
inline ScaledImage<T,WDimT,HDimT,V>
    make_scaled_image(const WDimT& wDim, const HDimT& hDim, const V<T>& data)
{
    return ScaledImage<T,WDimT,HDimT,V>(wDim, hDim, data);
}

template <typename Derived>
struct IsScaledImage {
    static constexpr bool value = std::is_base_of<
        ScaledImageExpression<Derived>, Derived>::value;
};

} //namespace rtac

template <class Derived> RTAC_HOSTDEVICE inline
rtac::DimIterator<Derived> operator+(int32_t offset, const rtac::DimIterator<Derived>& it)
{
    return rtac::DimIterator<Derived>(it) += offset;
}

#endif //_DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
