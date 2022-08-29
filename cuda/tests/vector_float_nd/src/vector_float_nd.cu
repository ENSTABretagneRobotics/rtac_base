#include "vector_float_nd.h"

constexpr unsigned int BlockSize = 1024;

__global__ void do_fill(VectorFloatNDView<float3> vect)
{
    for(auto tid = threadIdx.x; tid < vect.size(); tid += blockDim.x) {
        //vect.set(tid, float3{(float)tid,
        //                     100.0f + tid,
        //                     200.0f + tid});
        vect.set(tid, float3{0,0,0});
    }
}
void fill(VectorFloatNDView<float3> vect)
{
    do_fill<<<vect.size() / BlockSize + 1, BlockSize>>>(vect);
    cudaDeviceSynchronize();
}

__global__ void do_update(VectorFloatNDView<float3> vect)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(; tid < vect.size(); tid += blockDim.x*gridDim.x) {
        auto v = vect[tid];
        vect.set(tid, float3{v.x + 1, v.y + 1, v.z + 1});
    }
}
void update(VectorFloatNDView<float3> vect)
{
    do_update<<<vect.size() / BlockSize + 1, BlockSize>>>(vect);
    cudaDeviceSynchronize();
}

__global__ void do_update(rtac::types::VectorView<float3> vect)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(; tid < vect.size(); tid += blockDim.x*gridDim.x) {
        auto v = vect[tid];
        vect[tid] = float3{v.x + 1, v.y + 1, v.z + 1};
    }
}
void update(rtac::types::VectorView<float3> vect)
{
    do_update<<<vect.size() / BlockSize + 1, BlockSize>>>(vect);
    cudaDeviceSynchronize();
}


__global__ void do_update(VectorFloatNDView<float4> vect)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(; tid < vect.size(); tid += blockDim.x*gridDim.x) {
        auto v = vect[tid];
        vect.set(tid, float4{v.x + 1, v.y + 1, v.z + 1, v.w + 1});
    }
}
void update(VectorFloatNDView<float4> vect)
{
    do_update<<<vect.size() / BlockSize + 1, BlockSize>>>(vect);
    cudaDeviceSynchronize();
}


__global__ void do_update(rtac::types::VectorView<float4> vect)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    for(; tid < vect.size(); tid += blockDim.x*gridDim.x) {
        auto v = vect[tid];
        vect[tid] = float4{v.x + 1, v.y + 1, v.z + 1, v.w + 1};
    }
}
void update(rtac::types::VectorView<float4> vect)
{
    do_update<<<vect.size() / BlockSize + 1, BlockSize>>>(vect);
    cudaDeviceSynchronize();
}


