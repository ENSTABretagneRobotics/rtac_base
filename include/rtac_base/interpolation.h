#ifndef _DEF_RTAC_BASE_INTERPOLATION_H_
#define _DEF_RTAC_BASE_INTERPOLATION_H_

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <rtac_base/types/common.h>
#include <rtac_base/types/VectorView.h>

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

    using Indexes = rtac::types::Vector<unsigned int>;
    using Vector  = rtac::types::Vector<T>;
    //using Indexes = std::vector<unsigned int>;
    //using Vector  = std::vector<T>;

    using Xconst_iterator = typename Vector::const_iterator;

    protected:

    Vector x0_;
    Vector y0_;

    InterpolatorInterface(const rtac::types::VectorView<const T>& x0,
                          const rtac::types::VectorView<const T>& y0);

    public:

    const Vector& x0() const;
    const Vector& y0() const;
    
    unsigned int size() const;
    
    Xconst_iterator lower_bound(T x) const;
    std::vector<Xconst_iterator> lower_bound(const Vector& x) const;
    Indexes lower_bound_indexes(rtac::types::VectorView<const T> x) const;
    
    /**
     * Core interpolating method. To be reimplemented in subclasses.
     * 
     * @param x      values where to interpolate.
     * @param output matrix where to write the interpolated values.
     */
    virtual void interpolate(rtac::types::VectorView<const T> x,
                             rtac::types::VectorView<T> y) const = 0;
};

template <typename T>
class InterpolatorNearest;
template <typename T>
class InterpolatorLinear;
template <typename T>
class InterpolatorCubicSpline;

template <typename T>
class Interpolator
{
    public:
    
    using value_type = T;
    using Vector     = rtac::types::Vector<T>;

    enum Type {
        Nearest,
        Linear,
        CubicSpline
    };

    protected:

    typename InterpolatorInterface<T>::ConstPtr interpolator_;

    public:

    template <typename VectorT>
    static Interpolator<T> CreateNearest(const VectorT& x0, const VectorT& y0);
    template <typename VectorT>
    static Interpolator<T> CreateLinear(const VectorT& x0, const VectorT& y0);
    template <typename VectorT>
    static Interpolator<T> CreateCubicSpline(const VectorT& x0, const VectorT& y0);

    Interpolator(typename InterpolatorInterface<T>::ConstPtr interpolator);
    Interpolator(const rtac::types::VectorView<const T>& x0,
                 const rtac::types::VectorView<const T>& y0,
                 Type type = Nearest);
    template <typename VectorT>
    Interpolator(const VectorT& x0, const VectorT& y0, Type type = Nearest);

    const Vector& x0() const { return interpolator_->x0(); }
    const Vector& y0() const { return interpolator_->y0(); }
    
    unsigned int size() const { return interpolator_->size(); }
    
    void interpolate(rtac::types::VectorView<const T> x,
                     rtac::types::VectorView<T> y) const;
    template <typename VectorT>
    void interpolate(const VectorT& x, VectorT& y) const;
    template <typename VectorT>
    VectorT interpolate(const VectorT& x) const;
    template <typename VectorT>
    VectorT operator()(const VectorT& x) const;
};


/**
 * Nearest-Neighbor interpolator.
 */
template <typename T>
class InterpolatorNearest : public InterpolatorInterface<T>
{
    public:

    using Indexes = typename InterpolatorInterface<T>::Indexes;
    using Vector  = typename InterpolatorInterface<T>::Vector;

    public:

    InterpolatorNearest(const rtac::types::VectorView<const T>& x0,
                        const rtac::types::VectorView<const T>& y0);

    virtual void interpolate(rtac::types::VectorView<const T> x,
                             rtac::types::VectorView<T> y) const;
};

/**
 * Linear interpolator.
 */
template <typename T>
class InterpolatorLinear : public InterpolatorInterface<T>
{
    public:

    using Indexes = typename InterpolatorInterface<T>::Indexes;
    using Vector  = typename InterpolatorInterface<T>::Vector;

    public:

    InterpolatorLinear(const rtac::types::VectorView<const T>& x0,
                       const rtac::types::VectorView<const T>& y0);

    virtual void interpolate(rtac::types::VectorView<const T> x,
                             rtac::types::VectorView<T> y) const;
};

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

    InterpolatorCubicSpline(const rtac::types::VectorView<const T>& x0,
                            const rtac::types::VectorView<const T>& y0);

    virtual void interpolate(rtac::types::VectorView<const T> x,
                             rtac::types::VectorView<T>       y) const;
};

// Interpolator IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorInterface<T>::InterpolatorInterface(const rtac::types::VectorView<const T>& x0,
                              const rtac::types::VectorView<const T>& y0) :
    x0_(x0.size()), y0_(y0.size())
{
    assert(x0_.size() == y0_.size());
    std::memcpy(x0_.data(), x0.data(), sizeof(T)*x0.size());
    std::memcpy(y0_.data(), y0.data(), sizeof(T)*y0.size());
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
 * Retrieve iterator to the x0_ elements just below or equal to a value, for each value in
 * x.
 *
 * @return a vector of iterators pointing to valid values in x0_ below of equal
 * to x values (a std::range_error is throwed if am iterator is not valid).
 */
template <typename T>
std::vector<typename InterpolatorInterface<T>::Xconst_iterator> 
    InterpolatorInterface<T>::lower_bound(const Vector& x) const
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
typename InterpolatorInterface<T>::Indexes
    InterpolatorInterface<T>::lower_bound_indexes(rtac::types::VectorView<const T> x) const
{
    Indexes output(x.size());
    for(int i = 0; i < output.size(); i++) {
        output[i] = this->lower_bound(x[i]) - this->x0_.begin();
    }
    return output;
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateNearest(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(x0, y0, Nearest);
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateLinear(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(x0, y0, Linear);
}

template <typename T> template <typename VectorT>
Interpolator<T> Interpolator<T>::CreateCubicSpline(const VectorT& x0, const VectorT& y0)
{
    return Interpolator<T>(x0, y0, CubicSpline);
}

template <typename T> 
Interpolator<T>::Interpolator(typename InterpolatorInterface<T>::ConstPtr interpolator) :
    interpolator_(interpolator)
{}

template <typename T> 
Interpolator<T>::Interpolator(const rtac::types::VectorView<const T>& x0,
                              const rtac::types::VectorView<const T>& y0,
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
Interpolator<T>::Interpolator(const VectorT& x0, const VectorT& y0, Type type) :
    Interpolator<T>(rtac::types::VectorView<const T>(x0.size(), x0.data()),
                    rtac::types::VectorView<const T>(y0.size(), y0.data()),
                    type)
{}
    
template <typename T> 
void Interpolator<T>::interpolate(rtac::types::VectorView<const T> x,
                                  rtac::types::VectorView<T> y) const
{
    interpolator_->interpolate(x, y);
}

template <typename T> template <typename VectorT>
void Interpolator<T>::interpolate(const VectorT& x, VectorT& y) const
{
    interpolator_->interpolate(rtac::types::VectorView<const T>(x.size(), x.data()),
                               rtac::types::VectorView<T>(y.size(), y.data()));
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

// InterpolatorNearest IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorNearest<T>::InterpolatorNearest(const rtac::types::VectorView<const T>& x0,
                                            const rtac::types::VectorView<const T>& y0) :
    InterpolatorInterface<T>(x0, y0)
{}

template <typename T>
void InterpolatorNearest<T>::interpolate(rtac::types::VectorView<const T> x,
                                         rtac::types::VectorView<T> y) const
{
    using namespace rtac::types::indexing;

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

// InterpolatorLinear IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorLinear<T>::InterpolatorLinear(const rtac::types::VectorView<const T>& x0,
                                          const rtac::types::VectorView<const T>& y0) :
    InterpolatorInterface<T>(x0, y0)
{}

template <typename T>
void InterpolatorLinear<T>::interpolate(rtac::types::VectorView<const T> x,
                                        rtac::types::VectorView<T> y) const
{
    using namespace rtac::types::indexing;
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

// InterpolatorCubicSpline IMPLEMENTATION //////////////////////////////////////////
template <typename T>
InterpolatorCubicSpline<T>::InterpolatorCubicSpline(
                const rtac::types::VectorView<const T>& x0,
                const rtac::types::VectorView<const T>& y0) :
    InterpolatorInterface<T>(x0, y0)//,
    //a_(x0.size()-1), b_(x0.size()-1), c_(x0.size()-1), d_(x0.size()-1)
{
    this->load_coefs();
}

template <typename T>
void InterpolatorCubicSpline<T>::load_coefs()
{
    using namespace rtac::types::indexing;

    Eigen::Map<const Vector> x0(this->x0().data(), this->x0().size());
    Eigen::Map<const Vector> y0(this->y0().data(), this->y0().size());

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
void InterpolatorCubicSpline<T>::interpolate(rtac::types::VectorView<const T> x,
                                             rtac::types::VectorView<T> y) const
{
    using namespace rtac::types::indexing;
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

}; //namespace algorithm
}; //namespace rtac

#endif //_DEF_RTAC_BASE_INTERPOLATION_H_
