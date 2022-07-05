#ifndef _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
#define _DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_

#include <tuple>
#include <type_traits>

namespace rtac { namespace types {

template <class Derived, typename T>
struct ScaledDimension {
    T operator[](std::size idx) const {
        return reinterpret_cast<const Derived*>(this)->operator[](idx);
    }
};

template <class... DimTypes>
class ScaledDimension
{
    protected:

    std::tuple<DimTypes...> dimensions_;

    public:

    ScaledDimensions
};


template <typename T>
class LinearDimension : public ScaledDimension<LinearDimension<T>, T>
{
    protected:

    T a_;

    public:
    
    // Default constructor for vector compatibility
    LinearDimension(T a = 0) : a_(a) {}
    void set(T a) { a_ = a; }

    T operator[](std::size_t idx) const {
        return a_*idx
    }
};

template <typename T>
class AffineDimension : public ScaledDimension<AffineDimension<T>, T>
{
    protected:

    T a_;
    T b_;

    public:

    // Default constructor for vector compatibility
    AffineDimension(T a, T b) : a_(a), b_(b) {}
    void set(T a, T b) { a_ = a; b_ = b}

    AffineDimension(T min, T max, std::size_t size) { this->set(a,b,size); }
    void set(T min, T max, std::size_t size) {
        a_ = (max - min) / (size - 1);
        b_ = min;
    }

    T operator[](std::size_t idx) const {
        return a_*idx + b_;
    }
};

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_SCALED_ARRAY_H_
