#ifndef _DEF_RTAC_BASE_CONTAINERS_DIM_EXPRESSION_H_
#define _DEF_RTAC_BASE_CONTAINERS_DIM_EXPRESSION_H_

#include <cmath>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>

namespace rtac {

template <class Derived> class DimIterator;

/**
 * Abstract representation of a "dimension scale".
 *
 * This is intended to encode the sampling of a physical dimension of a data
 * array.
 *
 * For example, recorded sound is a one dimensional data array. The dimension
 * scale of the data array is time. One can use a DimExpression to encode the
 * timedate of each sample.
 *
 * In the context of the RTAC simulator, the main purpose of this type is to
 * encode both dimensions of a two-dimensional multi-beam sonar data. A
 * multi-beam sonar outputs a two dimensional array which dimensions are time
 * (or range) and direction. The direction might not be linearly sampled. The
 * DimExpression is a unifying interface allowing to implement both linear
 * sampling and arbitrary sampling in the most efficient way.
 * 
 * Implemented using the Curiously Recurring Template Pattern (CRTP) for static
 * polymorphism (this is to be seen as regular polymorphism but without using
 * virtual methods. The polymorphism is resolved at compile time and is
 * therefore usable in CUDA device code).
 */
template <class Derived>
struct DimExpression
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return static_cast<const Derived*>(this);
    }
    
    RTAC_HOSTDEVICE float operator[](uint32_t index) const { return this->index_to_value(index); }
    RTAC_HOSTDEVICE DimIterator<Derived> begin() const {
        return DimIterator<Derived>(this, 0);
    }
    RTAC_HOSTDEVICE DimIterator<Derived> end() const {
        return DimIterator<Derived>(this, this->size());
    }

    // these 3 method are to be reimplemented in subclasses
    RTAC_HOSTDEVICE uint32_t      size()   const { return this->cast()->size(); }
    RTAC_HOSTDEVICE Bounds<float> bounds() const { return this->cast()->bounds(); }
    RTAC_HOSTDEVICE float index_to_value(uint32_t index) const {
        return this->cast()->index_to_value(index);
    }

    RTAC_HOSTDEVICE float front() const { return this->operator[](0);                }
    RTAC_HOSTDEVICE float back()  const { return this->operator[](this->size() - 1); }
    
    // Leaving this might make usage more complicated.
    //RTAC_HOSTDEVICE uint32_t operator()(float value) const { return this->value_to_index(value); }
    //RTAC_HOSTDEVICE uint32_t value_to_index(float value) const {
    //    return this->cast()->value_to_index(value);
    //}
};

/**
 * This allows to iterate on the values of a DimExpression.
 */
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
struct IsDimExpression {
    static constexpr bool value = std::is_base_of<
        DimExpression<Derived>, Derived>::value;
};

/**
 * Encodes a linearly sampled dimension.
 */
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

    // the view for this type is itself. It can be easily copied
    RTAC_HOSTDEVICE LinearDim view() const { return *this; }
};

/**
 * Encodes an arbitrarily sampled dimension.
 */
template <template<typename> class VectorT>
class ArrayDim : public DimExpression< ArrayDim<VectorT> >
{
    protected:

    VectorT<float> data_;
    Bounds<float>  bounds_;
    
    public:

    template <template<typename> class VectorT2>
    RTAC_HOSTDEVICE ArrayDim(const VectorT2<float>& data, const Bounds<float>& bounds) :
        data_(data), bounds_(bounds)
    {}
    template <template<typename>class VectorT2>
    RTAC_HOSTDEVICE ArrayDim<VectorT>& operator=(const ArrayDim<VectorT2>& other) {
        data_   = other.data_;
        bounds_ = other.bounds_;
    }
    
    RTAC_HOSTDEVICE uint32_t      size()   const { return data_.size(); }
    RTAC_HOSTDEVICE Bounds<float> bounds() const { return bounds_;      }
    RTAC_HOSTDEVICE float index_to_value(uint32_t index) const { return data_[index]; }

    //RTAC_HOSTDEVICE ArrayDim<ConstVectorView> view() const {
    //    return ArrayDim<ConstVectorView>(ConstVectorView<float>(this->size(), data_.data()), 
    //                                     this->bounds());
    //}
    RTAC_HOSTDEVICE ArrayDim<ConstVectorView> view() const {
        return ArrayDim<ConstVectorView>(ConstVectorView<float>(this->size(), data_.data()), 
                                         this->bounds());
    }
};

template <template<typename> class VectorT>
inline ArrayDim<VectorT> make_array_dim(const VectorT<float>& data, const Bounds<float>& bounds)
{
    return ArrayDim<VectorT>(data, bounds);
}


} //namespace rtac

#endif //_DEF_RTAC_BASE_CONTAINERS_DIM_EXPRESSION_H_
