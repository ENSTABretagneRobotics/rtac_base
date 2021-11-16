/**
 * @file geometry.h
 */

#ifndef _DEF_RTAC_BASE_GEOMETRY_H_
#define _DEF_RTAC_BASE_GEOMETRY_H_

#include <rtac_base/types/common.h>

namespace rtac { namespace geometry {

using namespace rtac::types;

template <typename T>
constexpr T to_degrees(T radians)
{
    return radians * 180.0 / M_PI;
}

template <typename T>
constexpr T to_radians(T degrees)
{
    return degrees * M_PI / 180.0;
}

/**
 * Conversion from Tait-Bryan angles to Quaternion in NED convention.
 *
 * Input as radians.
 */
template <typename T>
Quaternion<T> quaternion_from_nautical_rad(T yaw, T pitch, T roll)
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
Quaternion<T> quaternion_from_nautical_degrees(T yaw, T pitch, T roll)
{
    return quaternion_from_nautical_rad(to_radians(yaw),
                                        to_radians(pitch),
                                        to_radians(roll));
}

/**
 * Find a vector non-colinear to v.
 *
 * The non-colinear vector is found by starting from a zero vector the same
 * size of v. Then a single coefficient is set to 1 at the same index as the
 * lowest coefficient of v.
 *
 * @return A vector non-colinear to v.
 */
template <typename T, int D>
Eigen::Matrix<T,D,1> find_noncolinear(const Eigen::Matrix<T,D,1>& v)
{
    Eigen::Matrix<T,D,1> res = v.cwiseAbs();
    int minIndex = 0;
    for(int i = 1; i < res.rows(); i++) {
        if(res(i) < res(minIndex)) {
            minIndex = i;
        }
    }
    res = Eigen::Matrix<T,D,1>::Zero();
    res(minIndex) = 1;
    return res;
}

/**
 * Find a vector orthogonal to v.
 *
 * An orthogonal vector is found by first finding a vector vn non-colinear to
 * v. Then an orthogonal vector to v is constructed by negating the projection
 * of vn on v.
 *
 * @return A vector orthogonal to v.
 */
template <typename T, int D>
Eigen::Matrix<T,D,1> find_orthogonal(const Eigen::Matrix<T,D,1>& v)
{
    Eigen::Matrix<T,D,1> res = find_noncolinear(v);
    Eigen::Matrix<T,D,1> vn = v.normalized();
    res = res - res.dot(vn) * vn;
    return res;
}

/**
 * Find the rotation matrix R the closest to M in the Frobenius norm. (M must
 * be close to be orthonormal. This is usefull to regularize rotation
 * matrices).
 *
 * The rotation matrix R is computed from the Singular Value Decomposition of
 * M ([See here](https://www.maths.manchester.ac.uk/~higham/narep/narep161.pdf)):
 *
 * \f[ M = U \Sigma V \f]
 * \f[ R = UV \f]
 *
 * @return A rotation matrix R closest to M in Frobenius norm.
 */
template <typename T, int D>
Eigen::Matrix<T,D,D> orthonormalized(const Eigen::Matrix<T,D,D>& M, T tol = 1e-6)
{
    // Produce a orthonormal matrix from m using SVD.
    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    // (closest orthonormal matrix in Frobenius norm ? (check it))
    Eigen::JacobiSVD<Eigen::Matrix<T,D,D>> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T,D,1> sv = svd.singularValues();
    if(sv(Eigen::last) < tol*sv(0)) {
        throw std::runtime_error(
            "Orthonormalized : bad conditionned matrix. Cannot orthonormalize.");
    }

    return svd.matrixU()*(svd.matrixV().transpose());
}

/**
 * See Pose::look_at for details.
 */
template <typename T>
Matrix3<T> look_at(const Vector3<T>& target, const Vector3<T>& position, const Vector3<T>& up)
{
    Matrix3<T> r;

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
        x = find_orthogonal(y);
    }
    x.normalize();
    Vector3<T> z = x.cross(y);

    r(all,0) = x; r(all,1) = y; r(all,2) = z;

    return r;
}

}; //namespace geometry
}; //namespace rtac


#endif //_DEF_RTAC_BASE_GEOMETRY_H_
