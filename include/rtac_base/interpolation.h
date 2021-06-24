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
    using Matrix  = types::Matrix<T>;

    using Xconst_iterator = typename Vector::const_iterator;

    protected:

    types::Handle<const Vector> x0_;
    types::Handle<const Matrix> y0_;

    Interpolator(const types::Handle<const Vector>& x0, const types::Handle<const Matrix>& y0);

    public:
    
    unsigned int output_dimension() const;
    Matrix operator()(const Vector& x) const;
    
    Xconst_iterator lower_bound(T x) const;
    template <class VectorT>
    std::vector<Xconst_iterator> lower_bound(const VectorT& x) const;

    /**
     * Core interpolating method. To be reimplemented in subclasses.
     * 
     * @param x      values where to interpolate.
     * @param output matrix where to write the interpolated values.
     */
    virtual void interpolate(const Vector& x, Matrix& output) const = 0;
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
    using Matrix  = typename Interpolator<T>::Matrix;

    public:

    InterpolatorNearest(const types::Handle<const Vector>& x0,
                        const types::Handle<const Matrix>& y0);
    InterpolatorNearest(const Vector& x0,
                        const Matrix& y0);

    virtual void interpolate(const Vector& x, Matrix& output) const;
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
    using Matrix  = typename Interpolator<T>::Matrix;

    public:

    InterpolatorLinear(const types::Handle<const Vector>& x0,
                        const types::Handle<const Matrix>& y0);
    InterpolatorLinear(const Vector& x0,
                        const Matrix& y0);

    virtual void interpolate(const Vector& x, Matrix& output) const;
};

// Interpolator IMPLEMENTATION //////////////////////////////////////////
template <typename T>
Interpolator<T>::Interpolator(const types::Handle<const Vector>& x0,
                              const types::Handle<const Matrix>& y0) :
    x0_(x0), y0_(y0)
{}

/**
 * Interpolate at values x.
 *
 * @param x values where to interpolate.
 *
 * @return Interpolated values.
 */
template <typename T>
typename Interpolator<T>::Matrix Interpolator<T>::operator()(const Vector& x) const
{
    Matrix output(x.rows(), this->output_dimension());
    this->interpolate(x, output);
    return output;
}

/**
 * Expected dimension of the interpolated values.
 *
 * (Interpolated values can be vectors).
 */
template <typename T>
unsigned int Interpolator<T>::output_dimension() const
{
    return y0_->cols();
}

/**
 * Find an iterator in x0_ the closest below or equal x.
 *
 * throws a std::range error if such iterator could not be found.
 */
template <typename T>
typename Interpolator<T>::Xconst_iterator Interpolator<T>::lower_bound(T x) const
{
    auto it = std::lower_bound(x0_->begin(), x0_->end(), x);
    if(it == x0_->end() || it == x0_->begin() && *it > x) {
        std::ostringstream oss;
        oss << "Iterator : a requested input value is not in input range ("
            << "range is [" << (*x0_)(0) << "-" << (*x0_)(Eigen::last)
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
template <typename T> template <class VectorT>
std::vector<typename Interpolator<T>::Xconst_iterator> 
    Interpolator<T>::lower_bound(const VectorT& x) const
{
    std::vector<Xconst_iterator> output(x.rows());
    for(int i = 0; i < output.size(); i++) {
        output[i] = this->lower_bound(x(i));
    }
    return output;
}

// InterpolatorNearest IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorNearest<T>::InterpolatorNearest(const types::Handle<const Vector>& x0,
                                            const types::Handle<const Matrix>& y0) :
    Interpolator<T>(x0, y0)
{}

template <typename T>
InterpolatorNearest<T>::InterpolatorNearest(const Vector& x0, const Matrix& y0) :
    InterpolatorNearest<T>(types::Handle<Vector>(new Vector(x0)),
                           types::Handle<Matrix>(new Matrix(y0)))
{}

template <typename T>
void InterpolatorNearest<T>::interpolate(const Vector& x, Matrix& output) const
{
    using namespace rtac::types::indexing;

    auto iterators = this->lower_bound(x);
    for(int i = 0; i < x.rows(); i++) {
        if(iterators[i] == this->x0_->end() - 1) {
            output(i,all) = (*this->y0_)(last, all);
            continue;
        }
        unsigned int idx = iterators[i] - this->x0_->begin();
        if(x(i) - (*this->x0_)(idx) <= (*this->x0_)(idx + 1) - x(i))
            output(i,all) = (*this->y0_)(idx, all);
        else
            output(i,all) = (*this->y0_)(idx + 1, all);
    }
}

// InterpolatorLinear IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorLinear<T>::InterpolatorLinear(const types::Handle<const Vector>& x0,
                                            const types::Handle<const Matrix>& y0) :
    Interpolator<T>(x0, y0)
{}

template <typename T>
InterpolatorLinear<T>::InterpolatorLinear(const Vector& x0, const Matrix& y0) :
    InterpolatorLinear<T>(types::Handle<Vector>(new Vector(x0)),
                           types::Handle<Matrix>(new Matrix(y0)))
{}

template <typename T>
void InterpolatorLinear<T>::interpolate(const Vector& x, Matrix& output) const
{
    using namespace rtac::types::indexing;
    auto iterators = this->lower_bound(x);
    for(int i = 0; i < x.rows(); i++) {
        if(iterators[i] == this->x0_->end() - 1) {
            output(i,all) = (*this->y0_)(last, all);
            continue;
        }
        unsigned int idx = iterators[i]  - this->x0_->begin();
        T lambda = (x(i) - (*this->x0_)(idx))
                 / ((*this->x0_)(idx + 1) - (*this->x0_)(idx));
        output(i,all) = (1.0 - lambda) * (*this->y0_)(idx,     all)
                      +        lambda  * (*this->y0_)(idx + 1, all);
    }
}

}; //namespace algorithm
}; //namespace rtac

#endif //_DEF_RTAC_BASE_INTERPOLATION_H_
