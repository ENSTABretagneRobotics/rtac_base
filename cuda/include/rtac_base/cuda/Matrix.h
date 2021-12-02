#ifndef _DEF_RTAC_BASE_CUDA_MATRIX_H_
#define _DEF_RTAC_BASE_CUDA_MATRIX_H_

#include <iostream>
#include <cassert>

/**
 * These classes provide a simple linear algebra interface to be used in CUDA
 * without heavy dependencies.
 *
 * They are not optimized for production and do not replace the performances of
 * a proper linear algebra library such Eigen.
 */
namespace rtac { namespace cuda { namespace linear {

template <unsigned int R, unsigned int C,
          unsigned int Rstride = C, unsigned int Cstride = 1>
struct StridesType {

    static constexpr unsigned int Rows      = R;
    static constexpr unsigned int Cols      = C;
    static constexpr unsigned int RowStride = Rstride;
    static constexpr unsigned int ColStride = Cstride;
    static constexpr unsigned int Size      = Rows*Cols;

    static constexpr unsigned int linear_index(unsigned int row, unsigned int col) {
        assert(row < Rows && col < Cols);
        return Rstride * row + Cstride * col;
    }

    template <unsigned int row, unsigned int col>
    struct LinearIndex {
        static_assert(row < Rows && col < Cols);
        static constexpr unsigned int value = Rstride * row + Cstride * col;
    };
};

template <typename T, class S, class D>
class MatrixBase {

    public:
    
    using value_type = T;
    using Strides    = S;
    using Derived    = D;

    static constexpr unsigned int Rows     = Strides::Rows;
    static constexpr unsigned int Cols     = Strides::Cols;
    static constexpr unsigned int Size     = Strides::Size;
    static constexpr unsigned int RowMajor = Strides::RowMajor;
    
    // protected:

    // T data_[Size];

    public:

    // MatrixBase();
    // MatrixBase(const MatrixBase<T,S>& other);
    
    template <class D2>
    MatrixBase<T,S,D>& operator=(const MatrixBase<T,S,D2>& other);

    T*       data();
    const T* data() const;

    T&       operator()(unsigned int row, unsigned int col);
    const T& operator()(unsigned int row, unsigned int col) const;
    
    template <unsigned int row, unsigned int col> T&       get();
    template <unsigned int row, unsigned int col> const T& get() const;

    T&       operator[](unsigned int idx);
    const T& operator[](unsigned int idx) const;

    constexpr unsigned int rows() const;
    constexpr unsigned int cols() const;
    constexpr unsigned int size() const;
};

template <typename T, class S>
class MatrixData : public MatrixBase<T, S, MatrixData<T, S>>
{
    public:

    using value_type = T;
    using Strides    = S;

    static constexpr unsigned int Rows     = Strides::Rows;
    static constexpr unsigned int Cols     = Strides::Cols;
    static constexpr unsigned int Size     = Strides::Size;
    static constexpr unsigned int RowMajor = Strides::RowMajor;

    protected:

    T data_[Size];

    public:

    MatrixData() {};

    T*       data()       { return data_; }
    const T* data() const { return data_; }
};

template <typename T, unsigned int R, unsigned int C>
using Matrix = MatrixData<T, StridesType<R,C>>;

template <typename T>
using Matrix3 = Matrix<T,3,3>;

/**
 * This is for Eigen-like initialization with operator<<.
 */
template <typename T, class S, class D, unsigned int N>
struct ProxyLoader {

    MatrixBase<T,S,D>* mat;

    ProxyLoader(MatrixBase<T,S,D>* m) : mat(m) {}

    MatrixBase<T,S,D>& finished() { return *mat; }

    template <typename T2>
    ProxyLoader<T,S,D,N+1> operator,(T2 value) {
        (*mat)[N] = value;
        return ProxyLoader<T,S,D,N+1>(mat);
    }
};

// IMPLEMENTATION ///////////////////////////////////

template <typename T, class S, class D> template <class D2>
MatrixBase<T,S,D>& MatrixBase<T,S,D>::operator=(const MatrixBase<T,S,D2>& other)
{
    for(int i = 0; i < Rows; i++) {
        for(int j = 0; j < Cols; j++) {
            (*this)(i,j) = other(i,j);
        }
    }
    return *this;
}

template <typename T, class S, class D>
T* MatrixBase<T,S,D>::data()
{
    return reinterpret_cast<D*>(this)->data();
}

template <typename T, class S, class D>
const T* MatrixBase<T,S,D>::data() const
{
    return reinterpret_cast<const D*>(this)->data();
}

template <typename T, class S, class D>
T& MatrixBase<T,S,D>::operator()(unsigned int row, unsigned int col)
{
    return this->data()[Strides::linear_index(row, col)];
}

template <typename T, class S, class D>
const T& MatrixBase<T,S,D>::operator()(unsigned int row, unsigned int col) const
{
    return this->data()[Strides::linear_index(row, col)];
}

template <typename T, class S, class D> template <unsigned int row, unsigned int col>
T& MatrixBase<T,S,D>::get()
{
    return this->data()[Strides::linear_index(row, col)];
}

template <typename T, class S, class D> template <unsigned int row, unsigned int col>
const T& MatrixBase<T,S,D>::get() const
{
    return this->data()[Strides::linear_index(row, col)];
}

template <typename T, class S, class D>
T& MatrixBase<T,S,D>::operator[](unsigned int idx)
{
    assert(idx < Size);
    return this->data()[idx];
}

template <typename T, class S, class D>
const T& MatrixBase<T,S,D>::operator[](unsigned int idx) const
{
    assert(idx < Size);
    return this->data()[idx];
}

template <typename T, class S, class D> constexpr 
unsigned int MatrixBase<T,S,D>::rows() const
{
    return Rows;
}

template <typename T, class S, class D> constexpr
unsigned int MatrixBase<T,S,D>::cols() const
{
    return Cols;
}

template <typename T, class S, class D> constexpr
unsigned int MatrixBase<T,S,D>::size() const
{
    return Size;
}

}; //namespace linear
}; //namespace cuda
}; //namespace rtac

template <typename T, class S, class D>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::linear::MatrixBase<T,S,D>& m)
{
    for(int i = 0; i < m.rows(); i++) {
        os << m(i,0);
        for(int j = 1; j < m.cols(); j++) {
            os << " " << m(i,j);
        }
        if(i < m.rows() - 1)
            os << std::endl;
    }
    return os;
}

template <typename T1, class S, class D, typename T2>
rtac::cuda::linear::ProxyLoader<T1,S,D,1> operator<<(
    rtac::cuda::linear::MatrixBase<T1,S,D>& mat, T2 value)
{
    mat[0] = value;
    return rtac::cuda::linear::ProxyLoader<T1,S,D,1>(&mat);
}

#endif //_DEF_RTAC_BASE_CUDA_MATRIX_H_



