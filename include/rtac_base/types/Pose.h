#ifndef _DEF_RTAC_BASE_TYPES_POSE_H_
#define _DEF_RTAC_BASE_TYPES_POSE_H_

#include <iostream>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/Exception.h>
#include <rtac_base/types/common.h>
#include <rtac_base/geometry.h>
#include <rtac_base/utilities.h>

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

    template <typename T2>
    RTAC_HOSTDEVICE Pose(const Pose<T2>& other) { *this = other; }
    template <typename T2>
    RTAC_HOSTDEVICE Pose<T>& operator=(const Pose<T2>& other) {
        r_ = other.r_.template cast<T>();
        t_ = other.t_.template cast<T>();
        return *this;
    }

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
    RTAC_HOSTDEVICE T& y()       { return t_(1); }
    RTAC_HOSTDEVICE T& z()       { return t_(2); }

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

    static Pose decode_string(const std::string& str,    char delimiter = ',');
    std::string encode_string(const std::string& format, char delimiter = ',');
};

template<typename T>template <class D0, class D1> RTAC_HOSTDEVICE inline
Pose<T> Pose<T>::from_rotation_matrix(const Eigen::DenseBase<D0>& r,
                                      const Eigen::DenseBase<D1>& t)
{ 
    return Pose<T>{r,t};
}

template <typename T> template <class D> RTAC_HOSTDEVICE inline
Pose<T> Pose<T>::from_translation(const Eigen::DenseBase<D>& t)
{
    return Pose<T>{Mat3::Identity(), t};
}

template <typename T> template <class D> RTAC_HOSTDEVICE inline
Pose<T> Pose<T>::from_homogeneous_matrix(const Eigen::DenseBase<D>& homogeneousMatrix)
{
    Pose res;
    res.r_ = homogeneousMatrix(Eigen::seqN(0,3), Eigen::seqN(0,3));
    res.t_ = homogeneousMatrix(Eigen::seqN(0,3), 3);
    return res;
}

template <typename T> template <class D> RTAC_HOSTDEVICE inline
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
template <typename T> RTAC_HOSTDEVICE inline
Pose<T>& Pose<T>::look_at(const Pose<T>::Vec3& target,
                          const Pose<T>::Vec3& position,
                          const Pose<T>::Vec3& up)
{
    *this = Pose<T>::from_rotation_matrix(geometry::look_at(target, position, up),
                                          position);
    return *this;
}

template <typename T> inline
Pose<T> Pose<T>::decode_string(const std::string& str, char delimiter)
{
    std::istringstream iss(str);
    std::string format;

    // getting format
    if(!std::getline(iss, format, delimiter)) {
        throw FormatError() << " : invalid pose string '" << str << "'";
    }
    
    // parsing depending on format
    if(format.find("quat") != std::string::npos) {
        // Pose encoded as a quaternion, format is tx,ty,tz,qw,qx,qy,qz
        auto values = parse_numbers<T,7>(iss, delimiter);
        return Pose<T>::from_quaternion(Quat(values[3],values[4],values[5],values[6]),
                                        Vec3(values[0],values[1],values[2])).normalize();
    }
    else if(format.find("hmat") != std::string::npos) {
        // Pose encoded as a quaternion, format is tx,ty,tz,qw,qx,qy,qz
        auto values = parse_numbers<T,12>(iss, delimiter);
        Mat4 hmat;
        hmat << values[0], values[1], values[2],  values[3], 
                values[4], values[5], values[6],  values[7], 
                values[8], values[9], values[10], values[11], 
                0.0, 0.0, 0.0, 1.0;
        return Pose<T>::from_homogeneous_matrix(hmat).normalize();
    }
    else {
        throw FormatError() << " : invalid pose string '" << str << "'";
        //T value;
        //try {
        //    value = std::stof(format);
        //}
        //catch(const std::invalid_argument&) {
        //    throw FormatError() << " : invalid pose string '" << str << "'";
        //}
        //auto values = parse_numbers<T>(iss, delimiter);
        //values.push_front(value);
        //switch(values.size()) {
        //    default: throw FormatError() << " : invalid pose string '" << str << "'"; break;
        //    case 7: 
        //}
    }
}

template <typename T> inline
std::string Pose<T>::encode_string(const std::string& format, char d)
{
    std::ostringstream oss;
    if(format == "quat") {
        auto q = this->quaternion();
        oss << "quat" << d;
        oss << this->x() << d << this->y() << d << this->z() << d
            << q.w() << d << q.x() << d << q.y() << d << q.z();
    }
    else if(format == "hmat") {
        oss << "hmat" << d;
        oss << r_(0,0) << d << r_(0,1) << d << r_(0,2) << d << t_(0) << d
            << r_(1,0) << d << r_(1,1) << d << r_(1,2) << d << t_(1) << d
            << r_(2,0) << d << r_(2,1) << d << r_(2,2) << d << t_(2) << d
            << 0.0 << d << 0.0 << d << 0.0 << d << 1.0;
    }
    else {
        throw FormatError() << " : unknown format '" << format << "'";
    }
    return oss.str();
}

}; // namespace rtac

template <typename T> inline
std::ostream& operator<<(std::ostream& os, const rtac::Pose<T>& pose) {
    os <<   "t : (" << pose.translation().transpose() << ")"
       << ", r : " << pose.quaternion();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POSE_H_

