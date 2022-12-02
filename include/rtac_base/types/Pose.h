#ifndef _DEF_RTAC_BASE_TYPES_POSE_H_
#define _DEF_RTAC_BASE_TYPES_POSE_H_

#include <iostream>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/common.h>
#include <rtac_base/geometry.h>

namespace rtac {

/**
 * Represent a full 3D pose (position and orientation).
 *
 * The Pose is represented by a 3D rotation matrix and a translation. 
 *
 * In this class (and in the RTAC framework in general), the right-hand
 * convension is used for homogeneous coordinates and matrices.
 *
 * \f[ \left[ \begin{array}{cc} y \\ 1 \end{array} \right] = 
 *                 \left[ \begin{array}{cc} R & T  \\ 
 *                 \mathtt O& 1 \end{array} \right] . 
 *                 \left[ \begin{array}{cc} x \\ 1 \end{array} \right] \f]
 *
 * @tparam T Base scalar type (float, double, ...)
*/
template <typename T>
struct Pose
{
    using Mat3 = Eigen::Matrix3<T>;
    using Vec3 = Eigen::Vector3<T>;
    using Mat4 = Eigen::Matrix4<T>;
    using Vec4 = Eigen::Vector4<T>;
    using Quat = Eigen::Quaternion<T>;

    Mat3 r_;
    Vec3 t_;

    RTAC_HOSTDEVICE Pose() : r_(Mat3::Identity()), t_(0,0,0) {}

    template <class D0, class D1> RTAC_HOSTDEVICE 
    Pose(const Eigen::DenseBase<D0>& r, const Eigen::DenseBase<D1>& t) : r_(r), t_(t) {}

    RTAC_HOSTDEVICE static Pose Identity() { return Pose(); }

    RTAC_HOSTDEVICE Pose& normalize(T tol = 1.0e-6) {
        r_ = geometry::orthonormalized(r_);
        return *this;
    }

    RTAC_HOSTDEVICE const Mat3& orientation()     const { return r_; }
    RTAC_HOSTDEVICE const Vec3& translation()     const { return t_; }
    RTAC_HOSTDEVICE const Mat3& rotation_matrix() const { return r_;  }
    RTAC_HOSTDEVICE Mat3&       orientation()           { return r_; }
    RTAC_HOSTDEVICE Vec3&       translation()           { return t_; }
    RTAC_HOSTDEVICE Mat3&       rotation_matrix()       { return r_;  }


    RTAC_HOSTDEVICE T  x() const { return t_(0); }
    RTAC_HOSTDEVICE T  y() const { return t_(1); }
    RTAC_HOSTDEVICE T  z() const { return t_(2); }
    RTAC_HOSTDEVICE T& x()       { return t_(0); }
    T& y()       { return t_(1); }
    T& z()       { return t_(2); }

    RTAC_HOSTDEVICE void set_orientation(const Mat3& r) { r_ = r; }
    RTAC_HOSTDEVICE void set_orientation(const Quat& q) { r_ = q.toRotationMatrix(); }
    RTAC_HOSTDEVICE void set_translation(const Vec3& t) { t_ = t; }

    RTAC_HOSTDEVICE Quat quaternion()         const { return Quat(r_); }
    RTAC_HOSTDEVICE Mat4 homogeneous_matrix() const {
        Mat4 res;
        res(Eigen::seqN(0,3), Eigen::seqN(0,3)) = r_;
        res(Eigen::seqN(0,3), 3)                = t_;
        res(3, Eigen::seqN(0,4)) << 0,0,0,1;
        return res;
    }

    RTAC_HOSTDEVICE Pose& invert() {
        t_ = r_.transpose()*t_;
        r_.transposeInPlace();
        return *this;
    }
    RTAC_HOSTDEVICE Pose inverse() const { 
        Pose inv;
        inv.t_ = r_.transpose()*t_;
        inv.r_ = r_.transpose();
        return inv;
    }

    RTAC_HOSTDEVICE Pose& operator*=(const Pose& rhs) {
        t_ = r_*rhs.t_ + t_;
        r_ = r_ * rhs.r_;
        return *this;
    }

    RTAC_HOSTDEVICE Pose operator*(const Pose& rhs) const { 
        Pose res = *this;
        return res *= rhs;
    }
    RTAC_HOSTDEVICE Mat3 operator*(const Mat3& rhs) const { return r_ * rhs;      }
    RTAC_HOSTDEVICE Vec3 operator*(const Vec3& rhs) const { return r_ * rhs + t_; }
    RTAC_HOSTDEVICE Vec4 operator*(const Vec4& rhs) const {
        Vec4 res;
        res(Eigen::seqN(0,3)) = r_*rhs(Eigen::seqN(0,3)) + rhs(3)*t_;
        return res;
    }

    RTAC_HOSTDEVICE Pose& look_at(const Vec3& target,
                                  const Vec3& position,
                                  const Vec3& up = Vec3({0,0,1}));
    RTAC_HOSTDEVICE T angle() const { return Eigen::AngleAxis<T>(r_).angle(); }
    

    // below a creation helpers
    template <class D0, class D1 = Vec3> RTAC_HOSTDEVICE 
    static Pose from_rotation_matrix(const Eigen::DenseBase<D0>& r,
                                     const Eigen::DenseBase<D1>& t = Vec3(0,0,0));

    template <class D> RTAC_HOSTDEVICE
    static Pose from_translation(const Eigen::DenseBase<D>& t);

    template <class D> RTAC_HOSTDEVICE
    static Pose from_homogeneous_matrix(const Eigen::DenseBase<D>& homogeneousMatrix);

    template <class D = Vec3> RTAC_HOSTDEVICE
    static Pose from_quaternion(const Quat& q, const Eigen::DenseBase<D>& t = Vec3(0,0,0));
};

template<typename T>template <class D0, class D1> RTAC_HOSTDEVICE 
Pose<T> Pose<T>::from_rotation_matrix(const Eigen::DenseBase<D0>& r,
                                      const Eigen::DenseBase<D1>& t)
{ 
    return Pose<T>{r,t};
}

template <typename T> template <class D> RTAC_HOSTDEVICE
Pose<T> Pose<T>::from_translation(const Eigen::DenseBase<D>& t)
{
    return Pose<T>{Mat3::Identity(), t};
}

template <typename T> template <class D> RTAC_HOSTDEVICE
Pose<T> Pose<T>::from_homogeneous_matrix(const Eigen::DenseBase<D>& homogeneousMatrix)
{
    Pose res;
    res.r_ = homogeneousMatrix(Eigen::seqN(0,3), Eigen::seqN(0,3));
    res.t_ = homogeneousMatrix(Eigen::seqN(0,3), 3);
    return res;
}

template <typename T> template <class D> RTAC_HOSTDEVICE
Pose<T> Pose<T>::from_quaternion(const Quat& q, const Eigen::DenseBase<D>& t)
{
    return Pose{Mat3(q), t};
}

/**
 * Makes the Pose "look at" a target.
 *
 * The translation_ component is set to position. The orientation_ is
 * calculated in such a way that the y vector of the pose will points to
 * target, and the z vector will mostly points towards up (y,z and up end up in
 * the same plane).
 *
 * This function is made after the
 * [gluLookAt](https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluLookAt.xml)
 * function, now deprecated from the OpenGL specification. It is most usefull
 * to position a camera on a scene for rendering or sensor simulation.
 *
 * @param target   The target to "look_at" (back of screen for 3D rendering).
 * @param position The new position of this Pose.
 * @param up       The top direction (top of screen for 3D rendering).
 */
template <typename T> RTAC_HOSTDEVICE
Pose<T>& Pose<T>::look_at(const Pose<T>::Vec3& target,
                          const Pose<T>::Vec3& position,
                          const Pose<T>::Vec3& up)
{
    *this = Pose<T>::from_rotation_matrix(geometry::look_at(target, position, up),
                                          position);
    return *this;
}

}; // namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Pose<T>& pose) {
    os <<   "t : (" << pose.translation().transpose() << ")"
       << ", r : " << pose.quaternion();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POSE_H_

