#ifndef _DEF_RTAC_BASE_TYPES_ARRAY_2D_H_
#define _DEF_RTAC_BASE_TYPES_ARRAY_2D_H_

#include <array>
#include <rtac_base/types/VectorView.h>

namespace rtac { namespace types {

template <typename T, template<typename>class VectorT>
class Array2D : protected VectorT<T>
{
    public:

    using value_type = T;
    using VectorType = VectorT<T>;

    protected:
    
    std::size_t rows_;
    std::size_t cols_;

    public:

    template <typename... Args>
    Array2D(std::size_t rows, std::size_t cols, Args... args) :
        rows_(rows), cols_(cols), VectorType(args...)
    {}

    RTAC_HOSTDEVICE std::size_t rows() const { return rows_; }
    RTAC_HOSTDEVICE std::size_t cols() const { return cols_; }
    RTAC_HOSTDEVICE std::size_t size() const { return rows_*cols_; }
    
    RTAC_HOSTDEVICE T operator()(std::size_t row, std::size_t col) const {
        return this->VectorType::operator[](cols_*row + col);
    }
    RTAC_HOSTDEVICE T& operator()(std::size_t row, std::size_t col) {
        return this->VectorType::operator[](cols_*row + col);
    }

    RTAC_HOSTDEVICE T operator[](std::size_t idx) const {
        return this->VectorType::operator[](idx);
    }
    RTAC_HOSTDEVICE T& operator[](std::size_t idx) {
        return this->VectorType::operator[](idx);
    }
    
    RTAC_HOSTDEVICE const T* data() const { return this->VectorType::data(); }
    RTAC_HOSTDEVICE T* data() { return this->VectorType::data(); }

    RTAC_HOSTDEVICE Array2D<const T,VectorView> view() const {
        return Array2D<const T,VectorView>{rows_, cols_, this->size(), this->data()};
    }
    RTAC_HOSTDEVICE Array2D<T,VectorView> view() {
        return Array2D<T,VectorView>{rows_, cols_, this->size(), this->data()};
    }
};

template <typename T, template<typename>class VectorT, typename... Args>
Array2D<T,VectorT> make_Array2D(std::size_t rows, std::size_t cols, Args... args)
{
    return Array2D<T,VectorT>{rows, cols, args...};
}


}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_ARRAY_2D_H_
