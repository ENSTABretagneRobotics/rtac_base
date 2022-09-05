#ifndef _DEF_RTAC_CUDA_VEC_MATH_H_
#define _DEF_RTAC_CUDA_VEC_MATH_H_

/**
 * The rational behind this file is that mathematical operations with vector
 * types are not part of CUDA standart (to keep compatibility with pure C
 * probably).  Some similar files were distributed as part of SDK but were
 * removed in recent releases. This file is therefore needed for the sake of
 * stability.
 */

#include <rtac_base/cuda_defines.h>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda { namespace vec_math {

RTAC_HOSTDEVICE inline float2 operator+(const float2& lhs, const float2& rhs) { return float2{lhs.x + rhs.x, lhs.y + rhs.y}; }
RTAC_HOSTDEVICE inline float2 operator-(const float2& lhs, const float2& rhs) { return float2{lhs.x - rhs.x, lhs.y - rhs.y}; }
RTAC_HOSTDEVICE inline float2 operator+(const float2& v, float a)             { return float2{v.x + a, v.y + a}; }
RTAC_HOSTDEVICE inline float2 operator-(const float2& v, float a)             { return float2{v.x - a, v.y - a}; }
RTAC_HOSTDEVICE inline float2 operator*(const float2& v, float a)             { return float2{v.x * a, v.y * a}; }
RTAC_HOSTDEVICE inline float2 operator/(const float2& v, float a)             { return float2{v.x / a, v.y / a}; }
RTAC_HOSTDEVICE inline float2 operator+(float a, const float2& v)             { return v + a; }
RTAC_HOSTDEVICE inline float2 operator*(float a, const float2& v)             { return v * a; }
RTAC_HOSTDEVICE inline float  dot(const float2& lhs, const float2& rhs)       { return lhs.x*rhs.x + lhs.y*rhs.y; }
RTAC_HOSTDEVICE inline float  norm2     (const float2& v)                     { return dot(v,v);       }
RTAC_HOSTDEVICE inline float  norm      (const float2& v)                     { return sqrt(dot(v,v)); }
RTAC_HOSTDEVICE inline float2 normalized(const float2& v)                     { return v / norm(v);    }
RTAC_HOSTDEVICE inline float  cross(const float2& lhs, const float2& rhs)     { return lhs.x*rhs.y - lhs.y*rhs.x; }



RTAC_HOSTDEVICE inline float3 operator+(const float3& lhs, const float3& rhs) { return float3{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }
RTAC_HOSTDEVICE inline float3 operator-(const float3& lhs, const float3& rhs) { return float3{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}; }
RTAC_HOSTDEVICE inline float3 operator+(const float3& v, float a)             { return float3{v.x + a, v.y + a, v.z + a}; }
RTAC_HOSTDEVICE inline float3 operator-(const float3& v, float a)             { return float3{v.x - a, v.y - a, v.z - a}; }
RTAC_HOSTDEVICE inline float3 operator*(const float3& v, float a)             { return float3{v.x * a, v.y * a, v.z * a}; }
RTAC_HOSTDEVICE inline float3 operator/(const float3& v, float a)             { return float3{v.x / a, v.y / a, v.z / a}; }
RTAC_HOSTDEVICE inline float3 operator+(float a, const float3& v)             { return v + a; }
RTAC_HOSTDEVICE inline float3 operator*(float a, const float3& v)             { return v * a; }
RTAC_HOSTDEVICE inline float  dot(const float3& lhs, const float3& rhs)       { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z; }
RTAC_HOSTDEVICE inline float  norm2     (const float3& v)                     { return dot(v,v);       }
RTAC_HOSTDEVICE inline float  norm      (const float3& v)                     { return sqrt(dot(v,v)); }
RTAC_HOSTDEVICE inline float3 normalized(const float3& v)                     { return v / norm(v);    }
RTAC_HOSTDEVICE inline float  cross(const float3& lhs, const float3& rhs)     { return float3{lhs.y*rhs.z - lhs.z*rhs.y,
                                                                                              lhs.z*rhs.x - lhs.x*rhs.z,
                                                                                              lhs.x*rhs.y - lhs.y*rhs.x}; }

RTAC_HOSTDEVICE inline float4 operator+(const float4& lhs, const float4& rhs) { return float4{lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w}; }
RTAC_HOSTDEVICE inline float4 operator-(const float4& lhs, const float4& rhs) { return float4{lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w}; }
RTAC_HOSTDEVICE inline float4 operator+(const float4& v, float a)             { return float4{v.x + a, v.y + a, v.z + a, v.w + a}; }
RTAC_HOSTDEVICE inline float4 operator-(const float4& v, float a)             { return float4{v.x - a, v.y - a, v.z - a, v.w - a}; }
RTAC_HOSTDEVICE inline float4 operator*(const float4& v, float a)             { return float4{v.x * a, v.y * a, v.z * a, v.w * a}; }
RTAC_HOSTDEVICE inline float4 operator/(const float4& v, float a)             { return float4{v.x / a, v.y / a, v.z / a, v.w / a}; }
RTAC_HOSTDEVICE inline float4 operator+(float a, const float4& v)             { return v + a; }
RTAC_HOSTDEVICE inline float4 operator*(float a, const float4& v)             { return v * a; }
RTAC_HOSTDEVICE inline float  dot(const float4& lhs, const float4& rhs)       { return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z + lhs.w*rhs.w; }
RTAC_HOSTDEVICE inline float  norm2     (const float4& v)                     { return dot(v,v);       }
RTAC_HOSTDEVICE inline float  norm      (const float4& v)                     { return sqrt(dot(v,v)); }
RTAC_HOSTDEVICE inline float4 normalized(const float4& v)                     { return v / norm(v);    }

} //namespace vec_math
} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_CUDA_VEC_MATH_H_
