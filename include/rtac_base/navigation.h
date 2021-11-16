#ifndef _DEF_RTAC_BASE_NAVIGATION_H_
#define _DEF_RTAC_BASE_NAVIGATION_H_

#include <rtac_base/types/common.h>
#include <rtac_base/geometry.h>

namespace rtac { namespace navigation {

template <typename T>
Eigen::Matrix<T,3,3> ned_to_enu_matrix()
{
    return (Eigen::Matrix<T,3,3>() << 0,1,0,
                                      1,0,0,
                                      0,0,-1).finished();
}

template <typename T>
Eigen::Quaternion<T> ned_to_enu_quaternion()
{
    // 0.70710678118 is sqrt(2) / 2
    return Eigen::Quaternion<T>(0, 0.70710678118, 0.70710678118, 0);
}

/**
 * Conversion of a rotation matrix from North East Down (NED) convention to
 * East North Up convention.
 */
template <typename T>
Eigen::Matrix<T,3,3> ned_to_enu(const Eigen::Matrix<T,3,3>& nedRot)
{
    return ned_to_enu_matrix<T>()*nedRot;
}

/**
 * Conversion of a Quaternion from North East Down (NED) convention to
 * East North Up convention.
 */
template <typename T>
Eigen::Quaternion<T> ned_to_enu(const Eigen::Quaternion<T>& nedRot)
{
    return ned_to_enu_quaternion<T>()*nedRot;
}

/**
 * Conversion from Tait-Bryan angles to Quaternion in NED convention.
 *
 * Input as radians.
 */
template <typename T>
Eigen::Quaternion<T> quaternion_from_nautical_rad(T yaw, T pitch, T roll)
{
    return Eigen::AngleAxis<T>(yaw,   Eigen::Vector3<T>::UnitZ())
         * Eigen::AngleAxis<T>(pitch, Eigen::Vector3<T>::UnitY())
         * Eigen::AngleAxis<T>(roll,  Eigen::Vector3<T>::UnitX());
}

/**
 * Conversion from Tait-Bryan angles to Quaternion in NED convention.
 *
 * Input as degrees.
 */
template <typename T>
Eigen::Quaternion<T> quaternion_from_nautical_degrees(T yaw, T pitch, T roll)
{
    return quaternion_from_nautical_rad(geometry::to_radians(yaw),
                                        geometry::to_radians(pitch),
                                        geometry::to_radians(roll));
}

}; //namespace navigation
}; //namespace rtac

#endif //_DEF_RTAC_BASE_NAVIGATION_H_
