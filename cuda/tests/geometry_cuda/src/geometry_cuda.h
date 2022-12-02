#pragma once

#include <rtac_base/cuda/geometry.h>
#include <rtac_base/types/common.h>
using namespace rtac;

float2 multiply(const Matrix2<float>& lhs, float2 rhs);
float2 multiply(float2 lhs, const Matrix2<float>& rhs);

float3 multiply(const Matrix3<float>& lhs, float3 rhs);
float3 multiply(float3 lhs, const Matrix3<float>& rhs);

float4 multiply(const Matrix4<float>& lhs, float4 rhs);
float4 multiply(float4 lhs, const Matrix4<float>& rhs);
