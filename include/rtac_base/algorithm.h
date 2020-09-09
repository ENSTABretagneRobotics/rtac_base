#ifndef _DEF_RTAC_BASE_ALGORITHM_H_
#define _DEF_RTAC_BASE_ALGORITHM_H_

#include <rtac_base/types/common.h>

namespace rtac { namespace algorithm {

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

}; //namespace algorithm
}; //namespace rtac
#endif //_DEF_RTAC_BASE_ALGORITHM_H_
