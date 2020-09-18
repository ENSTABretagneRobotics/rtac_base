#ifndef _DEF_RTAC_BASE_TYPES_POSE_H_
#define _DEF_RTAC_BASE_TYPES_POSE_H_

#include <iostream>

#include <rtac_base/types/common.h>
#include <rtac_base/algorithm.h>

namespace rtac { namespace types {

// class declaration
template <typename T>
class Pose
{
    protected:
    
    Vector3<T>    translation_;
    Quaternion<T> orientation_;

    public:

    static Pose<T> from_homogeneous_matrix(const Matrix4<T>& h);
    static Pose<T> from_rotation_matrix(const Matrix3<T>& r,
                                        const Vector3<T>& t = {0,0,0});
    
    Pose(const Vector3<T>&    translation = Vector3<T>(0,0,0),
         const Quaternion<T>& orientation = Quaternion<T>(1,0,0,0));

    void set_translation(const Vector3<T>& t);
    void set_orientation(const Quaternion<T>& q);
    void set_orientation(const Matrix3<T>& r);

    const Vector3<T>&    translation() const;
    const Quaternion<T>& orientation() const;

    Matrix3<T> rotation_matrix()    const;
    Matrix4<T> homogeneous_matrix() const;

    Pose<T>& operator*=(const Pose<T>& other);
    Pose<T>  inverse() const;
};

// class definition
template <typename T>
Pose<T>::Pose(const Vector3<T>& translation, const Quaternion<T>& orientation) :
    translation_(translation),
    orientation_(orientation)
{}

template <typename T>
Pose<T> Pose<T>::from_rotation_matrix(const Matrix3<T>& r,
                                      const Vector3<T>& t)
{
    return Pose<T>(t, Quaternion<T>(rtac::algorithm::orthonormalized(r)));
}

template <typename T>
Pose<T> Pose<T>::from_homogeneous_matrix(const Matrix4<T>& h)
{
    return from_rotation_matrix(h.block(0,0,3,3),
                                h(indexing::seqN(0,3), 3));
}

template <typename T>
void Pose<T>::set_translation(const Vector3<T>& t)
{
    translation_ = t;
}

template <typename T>
void Pose<T>::set_orientation(const Quaternion<T>& q)
{
    orientation_ = q.normalized();
}

template <typename T>
void Pose<T>::set_orientation(const Matrix3<T>& r)
{
    orientation_ = rtac::algorithm::orthonormalized(r);
}

template <typename T>
const Vector3<T>& Pose<T>::translation() const
{
    return translation_;
}

template <typename T>
const Quaternion<T>& Pose<T>::orientation() const
{
    return orientation_;
}

template <typename T>
Matrix3<T> Pose<T>::rotation_matrix() const
{
    return orientation_.toRotationMatrix();
}

template <typename T>
Matrix4<T> Pose<T>::homogeneous_matrix() const
{
    using namespace rtac::types::indexing;
    Matrix4<T> H;
    H(seq(0,2), seq(0,2)) = orientation_.toRotationMatrix();
    H(seq(0,2), last)     = translation_;
    H(3,0) = 0; H(3,1) = 0; H(3,2) = 0; H(3,3) = 1;
    return H;
}

template <typename T>
Pose<T>& Pose<T>::operator*=(const Pose<T>& other)
{
    translation_ = other.orientation_ * translation_ + other.translation_;
    orientation_ = other.orientation_ * orientation_;
}

template <typename T>
Pose<T> Pose<T>::inverse() const
{
    Quaternion<T> qinv = orientation_.inverse();
    return Pose<T>(-(qinv*translation_), qinv);
}

using Posef = Pose<float>;
using Posed = Pose<double>;

}; // namespace types
}; // namespace rtac

template<typename T>
rtac::types::Pose<T> operator*(const rtac::types::Pose<T>& lhs, const rtac::types::Pose<T>& rhs)
{
    rtac::types::Pose<T> res(rhs);
    res *= lhs;
    return res;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Pose<T>& pose) {
    os << "Pose"
       << ", t : (" << pose.translation().transpose() << ")"
       << ", r : " << pose.orientation();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POSE_H_

