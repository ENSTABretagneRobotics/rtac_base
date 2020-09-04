#ifndef _DEF_RTAC_TOOLS_TYPES_POSE_H_
#define _DEF_RTAC_TOOLS_TYPES_POSE_H_

#include <iostream>

#include <rtac_tools/types/common.h>

namespace rtac { namespace types {

// class declaration
template <typename T>
class Pose
{
    protected:
    
    Vector3<T>    translation_;
    Quaternion<T> orientation_;

    public:
    
    Pose(const Vector3<T>&    translation = Vector3<T>(0,0,0),
         const Quaternion<T>& orientation = Quaternion<T>(1,0,0,0));

    const Vector3<T>&    translation() const;
    const Quaternion<T>& orientation() const;

    Matrix3<T> rotation_matrix()    const;
    Matrix4<T> homogeneous_matrix() const;
};

// class definition
template <typename T>
Pose<T>::Pose(const Vector3<T>& translation, const Quaternion<T>& orientation) :
    translation_(translation),
    orientation_(orientation)
{}

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

using Posef = Pose<float>;
using Posed = Pose<double>;

}; // namespace types
}; // namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Pose<T>& pose) {
    os << "Pose"
       << ", t : (" << pose.translation().transpose() << ")"
       << ", r : " << pose.orientation();
    return os;
}

#endif //_DEF_RTAC_TOOLS_TYPES_POSE_H_

