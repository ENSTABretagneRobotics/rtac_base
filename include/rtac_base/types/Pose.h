#ifndef _DEF_RTAC_BASE_TYPES_POSE_H_
#define _DEF_RTAC_BASE_TYPES_POSE_H_

#include <iostream>

#include <rtac_base/types/common.h>
#include <rtac_base/geometry.h>

namespace rtac { namespace types {

// class declaration
template <typename T>
class Pose
{
    public:

    using Vec3       = rtac::types::Vector3<T>;
    using Quaternion = rtac::types::Quaternion<T>;
    using Mat3       = rtac::types::Matrix3<T>;
    using Mat4       = rtac::types::Matrix4<T>;

    protected:
    
    Vec3 translation_;
    Quaternion orientation_;

    public:

    static Pose<T> from_homogeneous_matrix(const Mat4& h);
    static Pose<T> from_rotation_matrix(const Mat3& r,
                                        const Vec3& t = {0,0,0});
    
    Pose(const Vec3& translation = Vec3(0,0,0),
         const Quaternion& orientation = Quaternion(1,0,0,0));

    void set_translation(const Vec3& t);
    void set_orientation(const Quaternion& q);
    void set_orientation(const Mat3& r);

    const Vec3&       translation() const;
    const Quaternion& orientation() const;

    Vec3&       translation();
    Quaternion& orientation();

    Matrix3<T> rotation_matrix()    const;
    Matrix4<T> homogeneous_matrix() const;

    Pose<T>& operator*=(const Pose<T>& rhs);
    Pose<T>  inverse() const;

    Pose<T>& look_at(const Vector3<T>& target,
                     const Vector3<T>& position,
                     const Vector3<T>& up = Vector3<T>({0,0,1}));

    T angle() const;

    T x() const { return translation_(0); }
    T y() const { return translation_(1); }
    T z() const { return translation_(2); }

    T qw() const { return orientation_.w(); }
    T qx() const { return orientation_.x(); }
    T qy() const { return orientation_.y(); }
    T qz() const { return orientation_.z(); }
};

// class definition
template <typename T>
Pose<T>::Pose(const Vec3& translation, const Quaternion& orientation) :
    translation_(translation),
    orientation_(orientation)
{}

template <typename T>
Pose<T> Pose<T>::from_rotation_matrix(const Mat3& r,
                                      const Vec3& t)
{
    return Pose<T>(t, Quaternion(rtac::geometry::orthonormalized(r)));
}

template <typename T>
Pose<T> Pose<T>::from_homogeneous_matrix(const Mat4& h)
{
    return from_rotation_matrix(h.block(0,0,3,3),
                                h(indexing::seqN(0,3), 3));
}

template <typename T>
void Pose<T>::set_translation(const Vec3& t)
{
    translation_ = t;
}

template <typename T>
void Pose<T>::set_orientation(const Quaternion& q)
{
    orientation_ = q.normalized();
}

template <typename T>
void Pose<T>::set_orientation(const Mat3& r)
{
    orientation_ = rtac::geometry::orthonormalized(r);
}

template <typename T>
const typename Pose<T>::Vec3& Pose<T>::translation() const
{
    return translation_;
}

template <typename T>
const typename Pose<T>::Quaternion& Pose<T>::orientation() const
{
    return orientation_;
}

template <typename T>
typename Pose<T>::Vec3& Pose<T>::translation()
{
    return translation_;
}

template <typename T>
typename Pose<T>::Quaternion& Pose<T>::orientation()
{
    return orientation_;
}


template <typename T>
typename Pose<T>::Mat3 Pose<T>::rotation_matrix() const
{
    return orientation_.toRotationMatrix();
}

template <typename T>
typename Pose<T>::Mat4 Pose<T>::homogeneous_matrix() const
{
    using namespace rtac::types::indexing;
    Matrix4<T> H;
    H(seq(0,2), seq(0,2)) = orientation_.toRotationMatrix();
    H(seq(0,2), last)     = translation_;
    H(3,0) = 0; H(3,1) = 0; H(3,2) = 0; H(3,3) = 1;
    return H;
}

template <typename T>
Pose<T>& Pose<T>::operator*=(const Pose<T>& rhs)
{
    this->set_translation(this->orientation()*rhs.translation() + this->translation());
    this->set_orientation(this->orientation()*rhs.orientation());
    return *this;
}

template <typename T>
Pose<T> Pose<T>::inverse() const
{
    Quaternion qinv = orientation_.inverse();
    return Pose<T>(-(qinv*translation_), qinv);
}

template <typename T>
Pose<T>& Pose<T>::look_at(const Vector3<T>& target,
                          const Vector3<T>& position,
                          const Vector3<T>& up)
{
    *this = Pose<T>::from_rotation_matrix(geometry::look_at(target, position, up),
                                          position);
    return *this;
}

template <typename T>
T Pose<T>::angle() const
{
    return std::asin(Vec3({orientation_.x(),
                           orientation_.y(),
                           orientation_.z()}).norm());
}

using Posef = Pose<float>;
using Posed = Pose<double>;

}; // namespace types
}; // namespace rtac

template<typename T>
rtac::types::Pose<T> operator*(const rtac::types::Pose<T>& lhs, 
                                 const rtac::types::Pose<T>& rhs)
{
    return rtac::types::Pose<T>(lhs.translation() + lhs.orientation() * rhs.translation(),
                                  lhs.orientation() * rhs.orientation());
}

template<typename T>
rtac::types::Pose<T> operator*(const rtac::types::Pose<T>& lhs,
                                 const typename rtac::types::Pose<T>::Quaternion& q)
{
    return rtac::types::Pose<T>(lhs.translation(), lhs.orientation() * q);
}

template<typename T>
rtac::types::Pose<T> operator*(const typename rtac::types::Pose<T>::Quaternion& q,
                                 const rtac::types::Pose<T>& rhs)
{
    return rtac::types::Pose<T>(q*rhs.translation(), q*rhs.orientation());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Pose<T>& pose) {
    os <<   "t : (" << pose.translation().transpose() << ")"
       << ", r : " << pose.orientation();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POSE_H_

