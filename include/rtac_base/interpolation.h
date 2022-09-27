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
template <typename T>
class Interpolator
{
    public:

    using Ptr      = types::Handle<Interpolator>;
    using ConstPtr = types::Handle<const Interpolator>;

    using Indexes = types::Vector<unsigned int>;
    using Vector  = types::Vector<T>;

    using Xconst_iterator = typename Vector::const_iterator;

    protected:

    Vector x0_;
    Vector y0_;

    Interpolator(const Vector& x0, const Vector& y0);

    public:

    const Vector& x0() const;
    const Vector& y0() const;
    
    Vector operator()(const Vector& x) const;
    unsigned int size() const;
    
    Xconst_iterator lower_bound(T x) const;
    std::vector<Xconst_iterator> lower_bound(const Vector& x) const;
    Indexes lower_bound_indexes(const Vector& x) const;

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
template <typename T>
class InterpolatorNearest : public Interpolator<T>
{
    public:

    using Indexes = typename Interpolator<T>::Indexes;
    using Vector  = typename Interpolator<T>::Vector;

    public:

    InterpolatorNearest(const Vector& x0, const Vector& y0);

    virtual void interpolate(const Vector& x, Vector& output) const;
};

/**
 * Linear interpolator.
 */
template <typename T>
class InterpolatorLinear : public Interpolator<T>
{
    public:

    using Indexes = typename Interpolator<T>::Indexes;
    using Vector  = typename Interpolator<T>::Vector;

    public:

    InterpolatorLinear(const Vector& x0, const Vector& y0);

    virtual void interpolate(const Vector& x, Vector& output) const;
};

/**
 * Cubic spline interpolator.
 *
 * y = an_.(x-xn)**3 + bn_.(x-xn)**2 + cn_.(x-xn) + dn_
 */
template <typename T>
class InterpolatorCubicSpline : public Interpolator<T>
{
    public:

    using Indexes = typename Interpolator<T>::Indexes;
    using Vector  = typename Interpolator<T>::Vector;

    protected:

    Vector a_;
    Vector b_;
    Vector c_;
    Vector d_;

    public:

    InterpolatorCubicSpline(const Vector& x0, const Vector& y0);

    virtual void interpolate(const Vector& x, Vector& output) const;
};

// Interpolator IMPLEMENTATION //////////////////////////////////////////
template <typename T>
Interpolator<T>::Interpolator(const Vector& x0, const Vector& y0) :
    x0_(x0), y0_(y0)
{
    assert(x0_.size() == y0_.size());
}

template <typename T>
const typename Interpolator<T>::Vector& Interpolator<T>::x0() const
{
    return x0_;
}

template <typename T>
const typename Interpolator<T>::Vector& Interpolator<T>::y0() const
{
    return y0_;
}

/**
 * Interpolate at values x.
 *
 * @param x values where to interpolate.
 *
 * @return Interpolated values.
 */
template <typename T>
typename Interpolator<T>::Vector Interpolator<T>::operator()(const Vector& x) const
{
    Vector output(x.size());
    this->interpolate(x, output);
    return output;
}

/**
 * @return number of data element which are interpolated (= size of origin data
 *         vectors)
 */
template <typename T>
unsigned int Interpolator<T>::size() const
{
    return x0_.size();
}

/**
 * Find an iterator in x0_ the closest below or equal x.
 *
 * throws a std::range error if such iterator could not be found.
 */
template <typename T>
typename Interpolator<T>::Xconst_iterator Interpolator<T>::lower_bound(T x) const
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
template <typename T>
std::vector<typename Interpolator<T>::Xconst_iterator> 
    Interpolator<T>::lower_bound(const Vector& x) const
{
    std::vector<Xconst_iterator> output(x.size());
    for(int i = 0; i < output.size(); i++) {
        output[i] = this->lower_bound(x[i]);
    }
    return output;
}

/**
 * Retrieve indexes to the x0_ elements just below or equal to a value, for
 * each value in x.
 *
 * @return a vector of indexes pointing values in x0_ below of equal to x
 * values (a std::range_error is throwed if an iterator is not valid).
 */
template <typename T>
typename Interpolator<T>::Indexes
    Interpolator<T>::lower_bound_indexes(const Vector& x) const
{
    Indexes output(x.size());
    for(int i = 0; i < output.size(); i++) {
        // not working 
        //output[i] = std::distance(this->lower_bound(x[i]), this->x0_.begin());
        output[i] = this->lower_bound(x[i]) - this->x0_.begin();
    }
    return output;
}

// InterpolatorNearest IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorNearest<T>::InterpolatorNearest(const Vector& x0, const Vector& y0) :
    Interpolator<T>(x0, y0)
{}

template <typename T>
void InterpolatorNearest<T>::interpolate(const Vector& x, Vector& output) const
{
    using namespace rtac::types::indexing;

    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            output[i] = this->y0_[idx[i]];
            continue;
        }
        if(x[i] - this->x0_[idx[i]] <= this->x0_[idx[i] + 1] - x[i])
            output[i] = this->y0_[idx[i]];
        else
            output[i] = this->y0_[idx[i] + 1];
    }
}

// InterpolatorLinear IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorLinear<T>::InterpolatorLinear(const Vector& x0, const Vector& y0) :
    Interpolator<T>(x0, y0)
{}

template <typename T>
void InterpolatorLinear<T>::interpolate(const Vector& x, Vector& output) const
{
    using namespace rtac::types::indexing;
    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            output[i] = this->y0_[idx[i]];
            continue;
        }
        T lambda = (x[i] - this->x0_[idx[i]])
                 / (this->x0_[idx[i] + 1] - this->x0_[idx[i]]);
        output[i] = (1.0 - lambda)*this->y0_[idx[i]] + lambda*this->y0_[idx[i] + 1];
    }
}

// InterpolatorCubicSpline IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorCubicSpline<T>::InterpolatorCubicSpline(const Vector& x0, const Vector& y0) :
    Interpolator<T>(x0, y0)//,
    //a_(x0.size()-1), b_(x0.size()-1), c_(x0.size()-1), d_(x0.size()-1)
{
    using namespace rtac::types::indexing;

    unsigned int size = x0.size();

    Vector dx =  x0(seqN(1,size-1)) - x0(seqN(0,size-1));
    Vector dy = (y0(seqN(1,size-1)) - y0(seqN(0,size-1))).array() / dx.array();

    Vector beta        =  6.0*(dy(seqN(1,dy.size()-1)) - dy(seqN(0,dy.size()-1)));
    types::Matrix<T> A = (2.0*(x0(seqN(2,x0.size()-2)) - x0(seqN(0,x0.size()-2)))).asDiagonal();
    for(int i = 0; i < this->size() - 3; i++) {
        A(i,i+1) = dx(i+1);
        A(i+1,i) = dx(i+1);
    }
    Vector alpha(this->size());
    alpha(seqN(1,alpha.size()-2)) = A.colPivHouseholderQr().solve(beta);
    alpha(0) = 0.0;
    alpha(alpha.size() - 1) = 0.0;
    
    this->a_ = (alpha(seqN(1,alpha.size()-1)) - alpha(seqN(0,alpha.size()-1))).array() / (6.0*dx.array());
    this->b_ = 0.5*alpha(seqN(1,alpha.size()-1));
    this->c_ = dy.array()
             + dx.array() * (2.0*alpha(seqN(1,alpha.size()-1)) + alpha(seqN(0,alpha.size()-1))).array() / 6.0;
    this->d_ = y0(seqN(1,y0.size()-1));
}

template <typename T>
void InterpolatorCubicSpline<T>::interpolate(const Vector& x, Vector& output) const
{
    using namespace rtac::types::indexing;
    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            output[i] = this->y0_[idx[i]];
            continue;
        }
        T v = x[i] - this->x0_[idx[i] + 1];
        output[i] = a_[idx[i]]*v*v*v + b_[idx[i]]*v*v + c_[idx[i]]*v + d_[idx[i]];
    }
}

}; //namespace algorithm
}; //namespace rtac

#endif //_DEF_RTAC_BASE_INTERPOLATION_H_
