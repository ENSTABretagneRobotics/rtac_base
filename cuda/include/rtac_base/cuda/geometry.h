#ifndef _DEF_RTAC_CUDA_GEOMETRY_H_
#define _DEF_RTAC_CUDA_GEOMETRY_H_

#include <rtac_base/cuda/vec_math.h>
#include <Eigen/Dense>

#include <rtac_base/types/common.h>
#include <rtac_base/geometry.h>
#include <rtac_base/types/Pose.h>

template <typename Derived> RTAC_HOSTDEVICE inline
float2 make_float2(const Eigen::MatrixBase<Derived>& other) {
    return float2{rtac::vector_get(other, 0),
                  rtac::vector_get(other, 1)};
}

template <typename Derived> RTAC_HOSTDEVICE inline
float3 make_float3(const Eigen::MatrixBase<Derived>& other) {
    return float3{rtac::vector_get(other, 0),
                  rtac::vector_get(other, 1),
                  rtac::vector_get(other, 2)};
}

template <typename Derived> RTAC_HOSTDEVICE inline
float4 make_float4(const Eigen::MatrixBase<Derived>& other) {
    return float4{rtac::vector_get(other, 0),
                  rtac::vector_get(other, 1),
                  rtac::vector_get(other, 2),
                  rtac::vector_get(other, 3)};
}


// These define common matrix operations between cuda vector types (float2/3/4)
// and corresponding Eigen fixed size matrices.
template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator*(const Eigen::MatrixBase<Derived>& lhs, const float2& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 2,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float2{(float)(lhs(0,0)*rhs.x + lhs(0,1)*rhs.y),
                  (float)(lhs(1,0)*rhs.x + lhs(1,1)*rhs.y)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator*(const Eigen::MatrixBase<Derived>& lhs, const float3& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 3,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float3{(float)(lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z),
                  (float)(lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z),
                  (float)(lhs(2,0)*rhs.x + lhs(2,1)*rhs.y + lhs(2,2)*rhs.z)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator*(const Eigen::MatrixBase<Derived>& lhs, const float4& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 4 && Derived::ColsAtCompileTime == 4,
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float4{(float)(lhs(0,0)*rhs.x + lhs(0,1)*rhs.y + lhs(0,2)*rhs.z + lhs(0,3)*rhs.w),
                  (float)(lhs(1,0)*rhs.x + lhs(1,1)*rhs.y + lhs(1,2)*rhs.z + lhs(1,3)*rhs.w),
                  (float)(lhs(2,0)*rhs.x + lhs(2,1)*rhs.y + lhs(2,2)*rhs.z + lhs(2,3)*rhs.w),
                  (float)(lhs(3,0)*rhs.x + lhs(3,1)*rhs.y + lhs(3,2)*rhs.z + lhs(3,3)*rhs.w)};
}
// vector*matrix multiplications are defined by transposition of the matrix*vector product
template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator*(const float2& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs.transpose()*lhs; }
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator*(const float3& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs.transpose()*lhs; }
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator*(const float4& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs.transpose()*lhs; }

// operator+ between eigen and cuda vector types
template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator+(const Eigen::MatrixBase<Derived>& lhs, const float2& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 2
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float2{(float)(lhs(0) + rhs.x),
                  (float)(lhs(1) + rhs.y)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator+(const Eigen::MatrixBase<Derived>& lhs, const float3& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 3
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float3{(float)(lhs(0) + rhs.x),
                  (float)(lhs(1) + rhs.y),
                  (float)(lhs(2) + rhs.z)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator+(const Eigen::MatrixBase<Derived>& lhs, const float4& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 4 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 4
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float4{(float)(lhs(0) + rhs.x),
                  (float)(lhs(1) + rhs.y),
                  (float)(lhs(2) + rhs.z),
                  (float)(lhs(3) + rhs.w)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator+(const float2& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs + lhs; }
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator+(const float3& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs + lhs; }
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator+(const float4& lhs, const Eigen::MatrixBase<Derived>& rhs) { return rhs + lhs; }

// operator- between eigen and cuda vector types
template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator-(const Eigen::MatrixBase<Derived>& lhs, const float2& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 2
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float2{(float)(lhs(0) - rhs.x),
                  (float)(lhs(1) - rhs.y)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator-(const Eigen::MatrixBase<Derived>& lhs, const float3& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 3
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float3{(float)(lhs(0) - rhs.x),
                  (float)(lhs(1) - rhs.y),
                  (float)(lhs(2) - rhs.z)};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator-(const Eigen::MatrixBase<Derived>& lhs, const float4& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 4 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 4
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float4{(float)(lhs(0) - rhs.x),
                  (float)(lhs(1) - rhs.y),
                  (float)(lhs(2) - rhs.z),
                  (float)(lhs(3) - rhs.w)};
}

template <typename Derived> RTAC_HOSTDEVICE inline
float2 operator-(const float2& lhs, const Eigen::MatrixBase<Derived>& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 2 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 2
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float2{(float)(lhs.x - rhs(0)),
                  (float)(lhs.y - rhs(1))};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float3 operator-(const float3& lhs, const Eigen::MatrixBase<Derived>& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 3
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float3{(float)(lhs.x - rhs(0)),
                  (float)(lhs.y - rhs(1)),
                  (float)(lhs.z - rhs(2))};
}
template <typename Derived> RTAC_HOSTDEVICE inline
float4 operator-(const float4& lhs, const Eigen::MatrixBase<Derived>& rhs) {
    EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived);
    EIGEN_STATIC_ASSERT(Derived::RowsAtCompileTime == 4 && Derived::ColsAtCompileTime == 1, 
                    ||  Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 4
                        THIS_METHOD_IS_ONLY_FOR_MATRICES_OF_A_SPECIFIC_SIZE);
    return float4{(float)(lhs.x - rhs(0)),
                  (float)(lhs.y - rhs(1)),
                  (float)(lhs.z - rhs(2)),
                  (float)(lhs.w - rhs(3))};
}



template <typename T> RTAC_HOSTDEVICE inline
float3 operator*(const rtac::Pose<T>& lhs, const float3& rhs) {
    return lhs.rotation_matrix()*rhs + lhs.translation();
}

#endif //_DEF_RTAC_CUDA_GEOMETRY_H_
