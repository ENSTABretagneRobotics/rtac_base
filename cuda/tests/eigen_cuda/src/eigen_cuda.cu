#include "eigen_cuda.h"

__global__ void do_multiply(Eigen::Matrix3f lhs, Eigen::Matrix3f rhs, Eigen::Matrix3f* dst)
{
    *dst = lhs * rhs;
}

Eigen::Matrix3f multiply(const Eigen::Matrix3f& lhs, const Eigen::Matrix3f& rhs)
{
    Eigen::Matrix3f res;

    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();

    return res;
}

__global__ void pose_mult(rtac::Pose<float> p, float3 in, float3* out)
{
    *out = p*in;
}
