#ifndef _DEF_RTAC_BASE_TYPES_COMMON_H_
#define _DEF_RTAC_BASE_TYPES_COMMON_H_

#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace rtac { namespace types {

template <class ... Types>
using MatrixBase = Eigen::MatrixBase<Types ...>;

template <class ... Types>
using Map = Eigen::Map<Types ...>;

template <typename T>
using Matrix  = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T>
using Matrix2  = Eigen::Matrix<T, 2, 2>;
template <typename T>
using Matrix3  = Eigen::Matrix<T, 3, 3>;
template <typename T>
using Matrix4  = Eigen::Matrix<T, 4, 4>;

template <typename T>
using Vector  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using Vector2  = Eigen::Matrix<T, 2, 1>;
template <typename T>
using Vector3  = Eigen::Matrix<T, 3, 1>;
template <typename T>
using Vector4  = Eigen::Matrix<T, 4, 1>;

template <typename T, size_t D>
class Array : public Eigen::Matrix<T, Eigen::Dynamic, D>
{
    public:

    Array() {};
    Array(size_t length) : Eigen::Matrix<T,Eigen::Dynamic,D>(length, D) {};
    Array(const Array<T,D>& other) :
        Eigen::Matrix<T,Eigen::Dynamic,D>(other) {};
    Array(const Eigen::Matrix<T,Eigen::Dynamic,D>& other) :
        Eigen::Matrix<T,Eigen::Dynamic,D>(other) {};
};
template <typename T>
using Array3  = Array<T,3>;

template <typename T>
using Quaternion  = Eigen::Quaternion<T>;

// Indexing and slicing aliases
// The namespace is to be able to use a 'using namespace rtac::types::indexing'
// without including all of the types.
namespace indexing {
    using Eigen::seq;
    using Eigen::seqN;
    using Eigen::last;
    using Eigen::lastN;
    using Eigen::all;
}

}; // namespace types
}; // namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Quaternion<T>& q) {
    os << "(" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << ")";
    return os;
}

void dummy();

#endif //_DEF_RTAC_BASE_TYPES_COMMON_H_
