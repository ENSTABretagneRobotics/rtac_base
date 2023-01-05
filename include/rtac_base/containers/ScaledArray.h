#ifndef _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
#define _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_

#include <cstdint>
#include <cmath>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/containers/Image.h>

namespace rtac {

template <class Derived> class DimIterator;

/**
 * Abstract representation of a single ScaledArray dimension.
 *
 * Implemented using the Curiously Recurring Template Pattern (CRTP) for static
 * polymorphism.
 *
 * The purpose of this object is to convert a physical dimension to an array
 * indice and the other way around.
 */
template <class Derived>
struct DimExpression
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return reinterpret_cast<const Derived*>(this);
    }
    
    RTAC_HOSTDEVICE float operator[](uint32_t index) const { return this->index_to_value(index); }
    RTAC_HOSTDEVICE DimIterator<Derived> begin() const { return DimIterator(this, 0); }
    RTAC_HOSTDEVICE DimIterator<Derived> end()   const { return DimIterator(this, this->size()); }

    // these 3 method are to be reimplemented in subclasses
    RTAC_HOSTDEVICE uint32_t      size()   const { return this->cast()->size(); }
    RTAC_HOSTDEVICE Bounds<float> bounds() const { return this->cast()->bounds(); }
    RTAC_HOSTDEVICE float index_to_value(uint32_t index) const {
        return this->cast()->index_to_value(index);
    }
    
    // Leaving this might make usage more complicated.
    //RTAC_HOSTDEVICE uint32_t operator()(float value) const { return this->value_to_index(value); }
    //RTAC_HOSTDEVICE uint32_t value_to_index(float value) const {
    //    return this->cast()->value_to_index(value);
    //}
};

class LinearDim : public DimExpression<LinearDim>
{
    protected:

    uint32_t      size_;
    Bounds<float> bounds_;
    float a_, b_;
    
    public:

    RTAC_HOSTDEVICE LinearDim(uint32_t size, const Bounds<float>& bounds) :
        size_(size), bounds_(bounds), a_(bounds.length() / (size - 1)), b_(bounds.lower)
    {}

    RTAC_HOSTDEVICE uint32_t      size()   const { return size_;   }
    RTAC_HOSTDEVICE Bounds<float> bounds() const { return bounds_; }
    RTAC_HOSTDEVICE float index_to_value(uint32_t index) const { return fmaf(a_, index, b_); }
};

template <class Derived>
class DimIterator
{
    protected:

    const DimExpression<Derived>* dim_;
    uint32_t index_;

    public:

    RTAC_HOSTDEVICE DimIterator(const DimExpression<Derived>* dim = nullptr, uint32_t index = 0) :
        dim_(dim), index_(index)
    {}

    RTAC_HOSTDEVICE DimIterator<Derived>& operator++() { index_++; return *this; }
    RTAC_HOSTDEVICE DimIterator<Derived>  operator++(int) const {
        return ++DimIterator<Derived>(*this);
    }
    RTAC_HOSTDEVICE DimIterator<Derived>& operator--() { index_--; return *this; }
    RTAC_HOSTDEVICE DimIterator<Derived>  operator--(int) const {
        return --DimIterator<Derived>(*this);
    }

    RTAC_HOSTDEVICE bool operator==(const DimIterator<Derived>& other) const {
        return index_ == other.index_ && dim_ == other.dim_;
    }
    RTAC_HOSTDEVICE bool operator!=(const DimIterator<Derived>& other) const {
        return !(*this == other);
    }

    RTAC_HOSTDEVICE float operator*() const { return (*dim_)[index_]; }
    RTAC_HOSTDEVICE float operator[](uint32_t offset) const { return (*dim_)[index_ + offset]; }

    RTAC_HOSTDEVICE DimIterator<Derived>& operator+=(int32_t offset) {
        index_ += offset; return *this;
    }
    RTAC_HOSTDEVICE DimIterator<Derived>& operator-=(int32_t offset) {
        index_ -= offset; return *this;
    }

    RTAC_HOSTDEVICE DimIterator<Derived> operator+(int32_t offset) const {
        return DimIterator<Derived>(*this) += offset;
    }
    RTAC_HOSTDEVICE DimIterator<Derived> operator-(int32_t offset) const {
        return DimIterator<Derived>(*this) -= offset;
    }
    RTAC_HOSTDEVICE int32_t operator-(const DimIterator<Derived>& other) const {
        return index_ - other.index;
    }
    
    RTAC_HOSTDEVICE bool operator< (const DimIterator<Derived>& other) const {
        return index_ <  other.offset_;
    }
    RTAC_HOSTDEVICE bool operator<=(const DimIterator<Derived>& other) const {
        return index_ <= other.offset_;
    }
    RTAC_HOSTDEVICE bool operator> (const DimIterator<Derived>& other) const {
        return other < *this;
    }
    RTAC_HOSTDEVICE bool operator>=(const DimIterator<Derived>& other) const {
        return other <= *this;
    }
};

template <class Derived>
struct ScaledArrayExpression
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return reinterpret_cast<const Derived*>(this);
    }
    RTAC_HOSTDEVICE Derived* cast() {
        return reinterpret_cast<Derived*>(this);
    }
    RTAC_HOSTDEVICE uint32_t width()  const { return this->width_dim().size();  }
    RTAC_HOSTDEVICE uint32_t height() const { return this->height_dim().size(); }

    //RTAC_HOSTDEVICE float width_dim_value(uint32_t index) const {
    //    return this->width_dim()(index);
    //}
    //RTAC_HOSTDEVICE float height_dim_value(uint32_t index) const {
    //    return this->height_dim()(index);
    //}

    RTAC_HOSTDEVICE const auto& width_dim()  const { return this->cast()->width_dim();  }
    RTAC_HOSTDEVICE const auto& height_dim() const { return this->cast()->height_dim(); }

    RTAC_HOSTDEVICE auto operator()(uint32_t h, uint32_t w) const {
        return (*this->cast())(h,w);
    }
    RTAC_HOSTDEVICE auto& operator()(uint32_t h, uint32_t w) {
        return (*this->cast())(h,w);
    }
};

template <class Derived>
struct IsDimExpression {
    static constexpr bool value = std::is_base_of<
        DimExpression<Derived>, Derived>::value;
};

template <class Derived>
struct IsScaledArray {
    static constexpr bool value = std::is_base_of<
        ScaledArrayExpression<Derived>, Derived>::value;
};

template <typename T,
          template<typename>class ContainerT,
          class WidthDimT,
          class HeightDimT>
struct ScaledImageConfig
{
    using value_type = T;
    using Container  = ContainerT<T>;
    using WidthDim   = WidthDimT;
    using HeightDim  = HeightDimT;
};

template <class ConfigT>
class ScaledImage
{
    public:
    
    using value_type = typename ConfigT::Container::value_type;
    using Container  = typename ConfigT::Container;
    using WidthDim   = typename ConfigT::WidthDim;
    using HeightDim  = typename ConfigT::HeightDim;
    using Shape      = rtac::Shape<uint32_t>;

    protected:

    Container data_;
    WidthDim  wDim_;
    HeightDim hDim_; 

    public:

    template <template<typename>class ContainerT>
    ScaledImage(const ContainerT<value_type>& data,
                const WidthDim& wDim,
                const HeightDim& hDim) :
        data_(data), wDim_(wDim), hDim_(hDim)
    {
        if(data_.size() != wDim_.size() * hDim_.size()) {
            throw std::runtime_error("Inconsistent sizes for ScaledImage");
        }
    }

    RTAC_HOSTDEVICE uint32_t width()  const { return wDim_.size(); }
    RTAC_HOSTDEVICE uint32_t height() const { return hDim_.size(); }
    RTAC_HOSTDEVICE Shape    shape()  const { return Shape(this->width(), this->height()); }
    RTAC_HOSTDEVICE auto     size()   const { return data_.size(); }

    RTAC_HOSTDEVICE value_type  operator[](std::size_t idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE value_type& operator[](std::size_t idx)       { return data_[idx]; }
    RTAC_HOSTDEVICE value_type  operator()(uint32_t h, uint32_t w) const { 
        return data_[this->width()*h + w];
    }
    RTAC_HOSTDEVICE value_type& operator()(uint32_t h, uint32_t w) { 
        return data_[this->width()*h + w];
    }
};

template <typename T, template<typename>class C, class WDimT, class HDimT>
inline ScaledImage<ScaledImageConfig<T,C,WDimT,HDimT>>
    make_scaled_image(const C<T>& data, const WDimT& wDim, const HDimT& hDim)
{
    return ScaledImage<ScaledImageConfig<T,C,WDimT,HDimT>>(data, wDim, hDim);
}
    

} //namespace rtac

template <class Derived> RTAC_HOSTDEVICE inline
rtac::DimIterator<Derived> operator+(int32_t offset, const rtac::DimIterator<Derived>& it)
{
    return rtac::DimIterator<Derived>(it) += offset;
}

#endif //_DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
