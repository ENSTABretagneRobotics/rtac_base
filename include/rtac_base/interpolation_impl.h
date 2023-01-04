#ifndef _DEF_RTAC_BASE_ALGORITHM_INTERPOLATION_IMPL_H_
#define _DEF_RTAC_BASE_ALGORITHM_INTERPOLATION_IMPL_H_

#include <memory>
#include <algorithm>

#include <rtac_base/types/common.h>
#include <rtac_base/containers/VectorView.h>

namespace rtac { namespace algorithm {
/**
 * Abstract base class representing a generic interpolator.
 */
template <typename T>
class InterpolatorInterface
{
    public:

    using Ptr      = std::shared_ptr<InterpolatorInterface>;
    using ConstPtr = std::shared_ptr<const InterpolatorInterface>;

    using Indexes = Eigen::VectorX<unsigned int>;
    using Vector  = Eigen::VectorX<T>;

    using Xconst_iterator = typename Vector::const_iterator;

    protected:

    Vector x0_;
    Vector y0_;

    InterpolatorInterface(VectorView<const T> x0,
                          VectorView<const T> y0);

    public:

    const Vector& x0() const;
    const Vector& y0() const;
    
    unsigned int size() const;
    
    Xconst_iterator lower_bound(T x) const;
    Indexes lower_bound_indexes(VectorView<const T> x) const;
    
    /**
     * Core interpolating method. To be reimplemented in subclasses.
     * 
     * @param x      values where to interpolate.
     * @param output matrix where to write the interpolated values.
     */
    virtual void interpolate(VectorView<const T> x,
                             VectorView<T> y) const = 0;
};

template <typename T>
InterpolatorInterface<T>::InterpolatorInterface(VectorView<const T> x0,
                                                VectorView<const T> y0) :
    x0_(Eigen::Map<const Vector>(x0.data(), x0.size())),
    y0_(Eigen::Map<const Vector>(y0.data(), y0.size()))
{
    assert(x0_.size() == y0_.size());
}

template <typename T>
const typename InterpolatorInterface<T>::Vector& InterpolatorInterface<T>::x0() const
{
    return x0_;
}

template <typename T>
const typename InterpolatorInterface<T>::Vector& InterpolatorInterface<T>::y0() const
{
    return y0_;
}

/**
 * @return number of data element which are interpolated (= size of origin data
 *         vectors)
 */
template <typename T>
unsigned int InterpolatorInterface<T>::size() const
{
    return x0_.size();
}

/**
 * Find an iterator in x0_ the closest below or equal x.
 *
 * throws a std::range error if such iterator could not be found.
 */
template <typename T>
typename InterpolatorInterface<T>::Xconst_iterator InterpolatorInterface<T>::lower_bound(T x) const
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
 * Retrieve indexes to the x0_ elements just below or equal to a value, for
 * each value in x.
 *
 * @return a vector of indexes pointing values in x0_ below of equal to x
 * values (a std::range_error is throwed if an iterator is not valid).
 */
template <typename T>
typename InterpolatorInterface<T>::Indexes
    InterpolatorInterface<T>::lower_bound_indexes(VectorView<const T> x) const
{
    Indexes output(x.size());
    for(int i = 0; i < output.size(); i++) {
        output[i] = this->lower_bound(x[i]) - this->x0_.begin();
    }
    return output;
}






/**
 * Nearest-Neighbor interpolator.
 */
template <typename T>
class InterpolatorNearest : public InterpolatorInterface<T>
{
    public:

    using Indexes = typename InterpolatorInterface<T>::Indexes;
    using Vector  = typename InterpolatorInterface<T>::Vector;

    InterpolatorNearest(VectorView<const T> x0,
                        VectorView<const T> y0) :
        InterpolatorInterface<T>(x0, y0)
    {}

    virtual void interpolate(VectorView<const T> x,
                             VectorView<T> y) const;
};

template <typename T>
void InterpolatorNearest<T>::interpolate(VectorView<const T> x,
                                         VectorView<T> y) const
{
    using namespace indexing;

    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            y[i] = this->y0_[idx[i]];
            continue;
        }
        if(x[i] - this->x0_[idx[i]] <= this->x0_[idx[i] + 1] - x[i])
            y[i] = this->y0_[idx[i]];
        else
            y[i] = this->y0_[idx[i] + 1];
    }
}


/**
 * Linear interpolator.
 */
template <typename T>
class InterpolatorLinear : public InterpolatorInterface<T>
{
    public:

    using Indexes = typename InterpolatorInterface<T>::Indexes;
    using Vector  = typename InterpolatorInterface<T>::Vector;

    InterpolatorLinear(VectorView<const T> x0,
                       VectorView<const T> y0) :
        InterpolatorInterface<T>(x0, y0)
    {}

    virtual void interpolate(VectorView<const T> x,
                             VectorView<T> y) const;
};

template <typename T>
void InterpolatorLinear<T>::interpolate(VectorView<const T> x,
                                        VectorView<T> y) const
{
    using namespace indexing;
    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            y[i] = this->y0_[idx[i]];
            continue;
        }
        T lambda = (x[i] - this->x0_[idx[i]])
                 / (this->x0_[idx[i] + 1] - this->x0_[idx[i]]);
        y[i] = (1.0 - lambda)*this->y0_[idx[i]] + lambda*this->y0_[idx[i] + 1];
    }
}


/**
 * Cubic spline interpolator.
 *
 * y = an_.(x-xn)**3 + bn_.(x-xn)**2 + cn_.(x-xn) + dn_
 */
template <typename T>
class InterpolatorCubicSpline : public InterpolatorInterface<T>
{
    public:

    using Indexes = typename InterpolatorInterface<T>::Indexes;
    using Vector  = typename InterpolatorInterface<T>::Vector;

    protected:

    Vector a_;
    Vector b_;
    Vector c_;
    Vector d_;

    void load_coefs();

    public:

    InterpolatorCubicSpline(VectorView<const T> x0,
                            VectorView<const T> y0) :
        InterpolatorInterface<T>(x0, y0)
    {
        this->load_coefs();
    }

    virtual void interpolate(VectorView<const T> x,
                             VectorView<T>       y) const;
};

template <typename T>
void InterpolatorCubicSpline<T>::load_coefs()
{
    using namespace indexing;

    Eigen::Map<const Vector> x0(this->x0().data(), this->x0().size());
    Eigen::Map<const Vector> y0(this->y0().data(), this->y0().size());

    unsigned int size = x0.size();

    Vector dx =  x0(seqN(1,size-1)) - x0(seqN(0,size-1));
    Vector dy = (y0(seqN(1,size-1)) - y0(seqN(0,size-1))).array() / dx.array();

    Vector beta        =  6.0*(dy(seqN(1,dy.size()-1)) - dy(seqN(0,dy.size()-1)));
    rtac::Matrix<T> A = (2.0*(x0(seqN(2,x0.size()-2)) - x0(seqN(0,x0.size()-2)))).asDiagonal();
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
void InterpolatorCubicSpline<T>::interpolate(VectorView<const T> x,
                                             VectorView<T> y) const
{
    using namespace indexing;
    Indexes idx = this->lower_bound_indexes(x);
    for(int i = 0; i < x.size(); i++) {
        if(idx[i] == this->x0_.size() - 1) {
            y[i] = this->y0_[idx[i]];
            continue;
        }
        T v = x[i] - this->x0_[idx[i] + 1];
        y[i] = a_[idx[i]]*v*v*v + b_[idx[i]]*v*v + c_[idx[i]]*v + d_[idx[i]];
    }
}


} //namespace algorithm
} //namespace rtac

#endif //_DEF_RTAC_BASE_ALGORITHM_INTERPOLATION_IMPL_H_
