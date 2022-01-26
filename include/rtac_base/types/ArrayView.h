#ifndef _DEF_RTAC_BASE_TYPES_ARRAY_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_ARRAY_VIEW_H_

#include <type_traits>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types { namespace array {

template <typename T, unsigned int S>
class StackVector
{
    public:

    using value_type = typename std::remove_const<T>::type;

    protected:
    
    T data_[S];

    public:

    RTAC_HOSTDEVICE const value_type& operator[](unsigned int idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE       value_type& operator[](unsigned int idx)       { return data_[idx]; }
    
    RTAC_HOSTDEVICE const value_type* data() const { return data_; }
    RTAC_HOSTDEVICE       value_type* data()       { return data_; }
};

template <typename T>
class ConstVectorView
{
    public:

    using value_type = typename std::remove_const<T>::type;

    protected:
    
    T* data_;

    public:

    ConstVectorView(T* data) : data_(data) {}

    RTAC_HOSTDEVICE const value_type& operator[](unsigned int idx) const { return data_[idx]; }
};

template <typename T>
class MutableVectorView : public ConstVectorView<T>
{
    public:

    using value_type = typename ConstVectorView<T>::value_type;

    public:

    MutableVectorView(T* data) : ConstVectorView<T>(data) {}

    RTAC_HOSTDEVICE value_type& operator[](unsigned int idx) { return this->data_[idx]; }
};

template <typename T>
class VectorView : public std::conditional<std::is_const<T>::value,
                                           ConstVectorView<T>,
                                           MutableVectorView<T>>::type
{
    public:
    
    using value_type     = T;
    using VectorViewBase = typename std::conditional<std::is_const<T>::value,
                                                     ConstVectorView<T>,
                                                     MutableVectorView<T>>::type;
    
    VectorView(T* data) : VectorViewBase(data) {}
};

template <unsigned int Stride>
struct StaticStride
{
    RTAC_HOSTDEVICE constexpr unsigned int outer_stride() const { return Stride; }
    RTAC_HOSTDEVICE constexpr unsigned int inner_stride() const { return 1; }

    RTAC_HOSTDEVICE constexpr unsigned int linear_index(unsigned int i, unsigned j) const {
        return this->outer_stride()*i + j;
    }
};

struct DynamicStride
{
    unsigned int outerStride_;

    RTAC_HOSTDEVICE           unsigned int outer_stride() const { return outerStride_; }
    RTAC_HOSTDEVICE constexpr unsigned int inner_stride() const { return 1; }

    RTAC_HOSTDEVICE unsigned int linear_index(unsigned int i, unsigned j) const {
        return this->outer_stride()*i + j;
    }
};

template <typename T, unsigned int R, unsigned int C>
class StackArray : public StackVector<T,R*C>, public StaticStride<C>
{
    public:

    using value_type = typename StackVector<T,R*C>::value_type;

    RTAC_HOSTDEVICE const T& operator()(unsigned int i, unsigned int j) const {
        return (*this)[this->linear_index(i,j)];
    }
    RTAC_HOSTDEVICE T& operator()(unsigned int i, unsigned int j) {
        return (*this)[this->linear_index(i,j)];
    }

    RTAC_HOSTDEVICE constexpr unsigned int rows() const { return R; }
    RTAC_HOSTDEVICE constexpr unsigned int cols() const { return C; }
};

template <typename T, unsigned int R, unsigned int C, unsigned int S = C>
class FixedArrayView : public VectorView<T>, public StaticStride<S>
{
    public:

    using value_type = typename StackVector<T,R*C>::value_type;

    FixedArrayView(T* data) : VectorView<T>(data) {}

    RTAC_HOSTDEVICE const T& operator()(unsigned int i, unsigned int j) const {
        return (*this)[this->linear_index(i,j)];
    }
    RTAC_HOSTDEVICE T& operator()(unsigned int i, unsigned int j) {
        return (*this)[this->linear_index(i,j)];
    }

    RTAC_HOSTDEVICE constexpr unsigned int rows() const { return R; }
    RTAC_HOSTDEVICE constexpr unsigned int cols() const { return C; }
};

template <typename T>
class ArrayView : public VectorView<T>, public DynamicStride
{
    public:

    using value_type = typename VectorView<T>::value_type;

    protected:

    unsigned int rows_;
    unsigned int cols_;

    public:

    ArrayView(unsigned int rows, unsigned int cols, T* data,
              unsigned int stride) : 
        VectorView<T>(data),
        DynamicStride({stride}),
        rows_(rows),
        cols_(cols)
    {}

    ArrayView(unsigned int rows, unsigned int cols, T* data) :
        ArrayView(rows, cols, data, cols)
    {}

    RTAC_HOSTDEVICE const T& operator()(unsigned int i, unsigned int j) const {
        return (*this)[this->linear_index(i,j)];
    }
    RTAC_HOSTDEVICE T& operator()(unsigned int i, unsigned int j) {
        return (*this)[this->linear_index(i,j)];
    }

    RTAC_HOSTDEVICE unsigned int rows() const { return rows_; }
    RTAC_HOSTDEVICE unsigned int cols() const { return cols_; }
};

}; //namespace array
}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_ARRAY_VIEW_H_
