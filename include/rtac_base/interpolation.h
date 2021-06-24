#ifndef _DEF_RTAC_BASE_INTERPOLATION_H_
#define _DEF_RTAC_BASE_INTERPOLATION_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <rtac_base/types/Handle.h>
#include <rtac_base/types/common.h>

namespace rtac { namespace algorithm {

/**
 * Abstract base class representing a generic interpolator.
 */
template <typename T, template<typename> class VectorT = std::vector>
class Interpolator
{
    public:

    using Ptr      = types::Handle<Interpolator>;
    using ConstPtr = types::Handle<const Interpolator>;

    using Indexes = VectorT<unsigned int>;
    using Vector  = VectorT<T>;

    using Xconst_iterator = typename Vector::const_iterator;

    protected:

    Vector x0_;
    Vector y0_;

    Interpolator(const Vector& x0, const Vector& y0);

    public:
    
    Vector operator()(const Vector& x) const;
    
    Xconst_iterator lower_bound(T x) const;
    std::vector<Xconst_iterator> lower_bound(const Vector& x) const;

    /**
     * Core interpolating method. To be reimplemented in subclasses.
     * 
     * @param x      values where to interpolate.
     * @param output matrix where to write the interpolated values.
     */
    virtual void interpolate(const Vector& x, Vector& output) const = 0;
};

/**
 * Nearest-Neighbor interpolator.
 */
template <typename T, template<typename>class VectorT = std::vector>
class InterpolatorNearest : public Interpolator<T,VectorT>
{
    public:

    using Indexes = typename Interpolator<T,VectorT>::Indexes;
    using Vector  = typename Interpolator<T,VectorT>::Vector;

    public:

    InterpolatorNearest(const Vector& x0, const Vector& y0);

    virtual void interpolate(const Vector& x, Vector& output) const;
};

/**
 * Linear interpolator.
 */
template <typename T, template<typename>class VectorT = std::vector>
class InterpolatorLinear : public Interpolator<T,VectorT>
{
    public:

    using Indexes = typename Interpolator<T,VectorT>::Indexes;
    using Vector  = typename Interpolator<T,VectorT>::Vector;

    public:

    InterpolatorLinear(const Vector& x0, const Vector& y0);

    virtual void interpolate(const Vector& x, Vector& output) const;
};

// Interpolator IMPLEMENTATION //////////////////////////////////////////
template <typename T, template<typename>class VectorT>
Interpolator<T,VectorT>::Interpolator(const Vector& x0, const Vector& y0) :
    x0_(x0), y0_(y0)
{}

/**
 * Interpolate at values x.
 *
 * @param x values where to interpolate.
 *
 * @return Interpolated values.
 */
template <typename T, template<typename>class VectorT>
typename Interpolator<T,VectorT>::Vector Interpolator<T,VectorT>::operator()(const Vector& x) const
{
    Vector output(x.size());
    this->interpolate(x, output);
    return output;
}

/**
 * Find an iterator in x0_ the closest below or equal x.
 *
 * throws a std::range error if such iterator could not be found.
 */
template <typename T, template<typename>class VectorT>
typename Interpolator<T,VectorT>::Xconst_iterator Interpolator<T,VectorT>::lower_bound(T x) const
{
    auto it = std::lower_bound(x0_.begin(), x0_.end(), x);
    if(it == x0_.end() || it == x0_.begin() && *it > x) {
        std::ostringstream oss;
        oss << "Iterator : a requested input value is not in input range ("
            << "range is [" << x0_[0] << "-" << *(x0_.end() - 1)
            << "], got " << x << ").";
        throw std::range_error(oss.str());
    }
    if(*it != x)
        it--;
    return it;
}

/**
 * Retrieve iterator to the x0_ elements just below or equal to a value, for each value in
 * x.
 *
 * @return a vector of iterators pointing to valid values in x0_ below of equal
 * to x values (a std::range_error is throwed if am iterator is not valid).
 */
template <typename T, template<typename>class VectorT>
std::vector<typename Interpolator<T,VectorT>::Xconst_iterator> 
    Interpolator<T,VectorT>::lower_bound(const Vector& x) const
{
    std::vector<Xconst_iterator> output(x.size());
    for(int i = 0; i < output.size(); i++) {
        output[i] = this->lower_bound(x[i]);
    }
    return output;
}

// InterpolatorNearest IMPLEMENTATION //////////////////////////////////////////
template <typename T, template<typename>class VectorT>
InterpolatorNearest<T,VectorT>::InterpolatorNearest(const Vector& x0, const Vector& y0) :
    Interpolator<T,VectorT>(x0, y0)
{}

template <typename T, template<typename>class VectorT>
void InterpolatorNearest<T,VectorT>::interpolate(const Vector& x, Vector& output) const
{
    using namespace rtac::types::indexing;

    auto iterators = this->lower_bound(x);
    for(int i = 0; i < x.size(); i++) {
        if(iterators[i] == this->x0_.end() - 1) {
            output[i] = *(this->y0_.end() - 1);
            continue;
        }
        unsigned int idx = iterators[i] - this->x0_.begin();
        if(x[i] - this->x0_[idx] <= this->x0_[idx + 1] - x[i])
            output[i] = this->y0_[idx];
        else
            output[i] = this->y0_[idx + 1];
    }
}

// InterpolatorLinear IMPLEMENTATION //////////////////////////////////////////
template <typename T, template<typename>class VectorT>
InterpolatorLinear<T,VectorT>::InterpolatorLinear(const Vector& x0, const Vector& y0) :
    Interpolator<T,VectorT>(x0, y0)
{}

template <typename T, template<typename>class VectorT>
void InterpolatorLinear<T,VectorT>::interpolate(const Vector& x, Vector& output) const
{
    using namespace rtac::types::indexing;
    auto iterators = this->lower_bound(x);
    for(int i = 0; i < x.size(); i++) {
        if(iterators[i] == this->x0_.end() - 1) {
            output[i] = *(this->y0_.end() - 1);
            continue;
        }
        unsigned int idx = iterators[i]  - this->x0_.begin();
        T lambda = (x[i] - this->x0_[idx])
                 / (this->x0_[idx + 1] - this->x0_[idx]);
        output[i] = (1.0 - lambda)*this->y0_[idx] + lambda*this->y0_[idx + 1];
    }
}

}; //namespace algorithm
}; //namespace rtac

#endif //_DEF_RTAC_BASE_INTERPOLATION_H_
