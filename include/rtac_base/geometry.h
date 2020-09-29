#ifndef _DEF_RTAC_BASE_GEOMETRY_H_
#define _DEF_RTAC_BASE_GEOMETRY_H_

#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>

namespace rtac { namespace geometry {

using namespace rtac::types;

// outputs a pose looking towards a point, assuming x-left, y-front, z-up local camera frame
template <typename T>
Pose<T> look_at(const Vector3<T>& target, const Vector3<T>& position, const Vector3<T>& up)
{
    using namespace rtac::types::indexing;
    // local y points towards target.
    Vector3<T> y = target - position;
    if(y.norm() < 1e-6) {
        // Camera too close to target, look towards world y.
        y = Vector3<T>({0.0,1.0,0.0});
    }
    y.normalize();

    Vector3<T> x = y.cross(up);
    if(x.norm() < 1e-6) {
        // No luck... We have to find another non-colinear vector.
        x = rtac::algorithm::find_orthogonal(y);
    }
    x.normalize();
    Vector3<T> z = x.cross(y);

    Matrix3<T> r;
    r(all,0) = x; r(all,1) = y; r(all,2) = z;

    return Pose<T>::from_rotation_matrix(r, position);
}

}; //namespace geometry
}; //namespace rtac


#endif //_DEF_RTAC_BASE_GEOMETRY_H_
