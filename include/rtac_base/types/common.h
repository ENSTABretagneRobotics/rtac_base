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
using Matrixf = Matrix<float>;
using Matrixd = Matrix<double>;

template <typename T>
using Matrix3  = Eigen::Matrix<T, 3, 3>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <typename T>
using Matrix4  = Eigen::Matrix<T, 4, 4>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <typename T>
using Vector  = Eigen::Matrix<T, Eigen::Dynamic, 1>;
using Vectorf = Vector<float>;
using Vectord = Vector<double>;

template <typename T>
using Vector3  = Eigen::Matrix<T, 3, 1>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

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
using Array3f = Array3<float>;
using Array3d = Array3<double>;

template <typename T>
using Quaternion  = Eigen::Quaternion<T>;
using Quaternionf = Quaternion<float>;
using Quaterniond = Quaternion<double>;

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

// Pointers definition (defined here to avoid cumbering the type definition).
template <typename T>
using MatrixPtr  = std::shared_ptr<Matrix<T>>;
using MatrixfPtr = std::shared_ptr<Matrixf>;
using MatrixdPtr = std::shared_ptr<Matrixd>;
template <typename T>
using MatrixConstPtr  = std::shared_ptr<const Matrix<T>>;
using MatrixfConstPtr = std::shared_ptr<const Matrixf>;
using MatrixdConstPtr = std::shared_ptr<const Matrixd>;


template <typename T, size_t D>
using ArrayPtr  = std::shared_ptr<Array<T,D>>;
template <typename T>
using Array3Ptr  = std::shared_ptr<Array3<T>>;
using Array3fPtr = std::shared_ptr<Array3f>;
using Array3dPtr = std::shared_ptr<Array3d>;
template <typename T, size_t D>
using ArrayConstPtr  = std::shared_ptr<const Array<T,D>>;
template <typename T>
using Array3ConstPtr  = std::shared_ptr<const Array3<T>>;
using Array3fConstPtr = std::shared_ptr<const Array3f>;
using Array3dConstPtr = std::shared_ptr<const Array3d>;

}; // namespace types
}; // namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Quaternion<T>& q) {
    os << "(" << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << ")";
    return os;
}

void dummy();

#endif //_DEF_RTAC_BASE_TYPES_COMMON_H_
