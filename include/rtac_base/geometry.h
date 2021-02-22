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

template <typename T, int D>
Eigen::Matrix<T,D,1> find_noncolinear(const Eigen::Matrix<T,D,1>& v, float tol = 1e-6)
{
    Eigen::Matrix<T,D,1> res = v.cwiseAbs();
    int maxIndex = 0;
    for(int i = 1; i < res.rows(); i++) {
        if(res(i) > res(maxIndex)) {
            maxIndex = i;
        }
    }
    res = v;
    res(maxIndex) = 0;
    if (res.norm() < tol) {
        // v is in canonical basis. Returning a different canonical vector.
        for(int i = 0; i < res.rows(); i++) {
            res(i) = 0;
        }
        res((maxIndex + 1) % res.rows()) = 1;
    }
    return res;
}

template <typename T, int D>
Eigen::Matrix<T,D,1> find_orthogonal(const Eigen::Matrix<T,D,1>& v, float tol = 1e-6)
{
    Eigen::Matrix<T,D,1> res = find_noncolinear(v, tol);
    Eigen::Matrix<T,D,1> vn = v.normalized();
    res = res - res.dot(vn) * vn;
    return res;
}

template <typename T, int D>
Eigen::Matrix<T,D,D> orthonormalized(const Eigen::Matrix<T,D,D>& m, T tol = 1e-6)
{
    // Produce a orthonormal matrix from m using SVD.
    // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    // (closest orthonormal matrix in Frobenius norm ? (check it))
    Eigen::JacobiSVD<Eigen::Matrix<T,D,D>> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<T,D,1> sv = svd.singularValues();
    if(sv(Eigen::last) < tol*sv(0))
        throw std::runtime_error("Orthonormalized : bad conditionned matrix. Cannot orthonormalize.");

    return svd.matrixU()*(svd.matrixV().transpose());
}

// outputs a pose looking towards a point, assuming x-left, y-front, z-up local camera frame
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
