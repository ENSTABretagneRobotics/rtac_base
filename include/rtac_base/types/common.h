#ifndef _DEF_RTAC_BASE_TYPES_COMMON_H_
#define _DEF_RTAC_BASE_TYPES_COMMON_H_

#include <iostream>
#include <memory>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace rtac {

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
using Matrix6  = Eigen::Matrix<T, 6, 6>;

template <typename T>
using Vector  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using Vector2  = Eigen::Matrix<T, 2, 1>;
template <typename T>
using Vector3  = Eigen::Matrix<T, 3, 1>;
template <typename T>
using Vector4  = Eigen::Matrix<T, 4, 1>;

template <typename T>
using Quaternion  = Eigen::Quaternion<T>;

// Indexing and slicing aliases
// The namespace is to be able to use a 'using namespace rtac::indexing'
// without including all of the types.
namespace indexing {
    using namespace Eigen::indexing;
}


}; // namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Quaternion<T>& q) {
    os << "(" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << ")";
    return os;
}

void dummy();

#endif //_DEF_RTAC_BASE_TYPES_COMMON_H_
