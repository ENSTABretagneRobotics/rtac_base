#include "geometry_cuda.h"

__global__ void do_multiply(Matrix2<float> lhs, float2 rhs, float2* dst)
{
    *dst = lhs*rhs;
}

float2 multiply(const Matrix2<float>& lhs, float2 rhs)
{
    float2 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}

__global__ void do_multiply(float2 lhs, Matrix2<float> rhs, float2* dst)
{
    *dst = lhs*rhs;
}

float2 multiply(float2 lhs, const Matrix2<float>& rhs)
{
    float2 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}


__global__ void do_multiply(Matrix3<float> lhs, float3 rhs, float3* dst)
{
    *dst = lhs*rhs;
}

float3 multiply(const Matrix3<float>& lhs, float3 rhs)
{
    float3 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}

__global__ void do_multiply(float3 lhs, Matrix3<float> rhs, float3* dst)
{
    *dst = lhs*rhs;
}

float3 multiply(float3 lhs, const Matrix3<float>& rhs)
{
    float3 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}


__global__ void do_multiply(Matrix4<float> lhs, float4 rhs, float4* dst)
{
    *dst = lhs*rhs;
}

float4 multiply(const Matrix4<float>& lhs, float4 rhs)
{
    float4 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}

__global__ void do_multiply(float4 lhs, Matrix4<float> rhs, float4* dst)
{
    *dst = lhs*rhs;
}

float4 multiply(float4 lhs, const Matrix4<float>& rhs)
{
    float4 res;
    do_multiply<<<1,1>>>(lhs, rhs, &res);
    cudaDeviceSynchronize();
    return res;
}


