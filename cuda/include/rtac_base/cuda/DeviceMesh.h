#ifndef _DEF_RTAC_CUDA_DEVICE_MESH_H_
#define _DEF_RTAC_CUDA_DEVICE_MESH_H_

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/types/Mesh.h>

#include <rtac_base/cuda/CudaVector.h>

namespace rtac { namespace cuda {

template <typename P = float3,
          typename F = uint3,
          typename N = float3,
          typename U = float2>
using DeviceMesh = rtac::Mesh<P,F,N,U,CudaVector>;

template <typename P = float3,
          typename F = uint3,
          typename N = float3,
          typename U = float2>
using HostMesh = rtac::Mesh<P,F,N,U,HostVector>;

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_DEVICE_MESH_H_
