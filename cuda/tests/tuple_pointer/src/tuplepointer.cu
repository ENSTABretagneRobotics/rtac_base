#include "tuplepointer.h"

__global__ void do_stuff_device(TuplePointer<float,uint16_t,size_t> t, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    std::get<0>(t[tid]) += 1;
    std::get<1>(t[tid]) += 1;
    std::get<2>(t[tid]) += 1;
}
void do_stuff(TuplePointer<float,uint16_t,size_t>& t, size_t size)
{
    do_stuff_device<<<(size / 512) + 1,512>>>(t, size);
    cudaDeviceSynchronize();
}

__global__ void do_stuff_array_device(float* p1, uint16_t* p2, size_t* p3, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    p1[tid] += 1;
    p2[tid] += 1;
    p3[tid] += 1;
}
void do_stuff_array(float* p1, uint16_t* p2, size_t* p3, size_t size)
{
    do_stuff_array_device<<<(size / 512) + 1,512>>>(p1, p2, p3, size);
    cudaDeviceSynchronize();
}

__global__ void do_stuff_struct_device(Data* data, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    data[tid].f1 += 1;
    data[tid].f2 += 1;
    data[tid].f3 += 1;
}

void do_stuff_struct(Data* data, size_t size)
{
    do_stuff_struct_device<<<(size / 512) + 1,512>>>(data, size);
    cudaDeviceSynchronize();
}


__global__ void do_stuff_device(TuplePointer<double,double,double,double> t, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    std::get<0>(t[tid]) += 1;
    std::get<1>(t[tid]) += 1;
    std::get<2>(t[tid]) += 1;
    std::get<3>(t[tid]) += 1;
}
void do_stuff(TuplePointer<double,double,double,double>& t, size_t size)
{
    do_stuff_device<<<(size / 512) + 1,512>>>(t, size);
    cudaDeviceSynchronize();
}

__global__ void do_stuff_array_device(double* f0, double* f1, double* f2, double* f3, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    f0[tid] += 1;
    f1[tid] += 1;
    f2[tid] += 1;
    f3[tid] += 1;
}
void do_stuff_array(double* f0, double* f1, double* f2, double* f3, size_t size)
{
    do_stuff_array_device<<<(size / 512) + 1,512>>>(f0, f1, f2, f3, size);
    cudaDeviceSynchronize();
}

__global__ void do_stuff_struct_device(Data2* data, size_t size)
{
    auto tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid > size) return;
    data[tid].f32[0] += 1;
    data[tid].f32[1] += 1;
    data[tid].f32[2] += 1;
    data[tid].f32[3] += 1;
}

void do_stuff_struct(Data2* data, size_t size)
{
    do_stuff_struct_device<<<(size / 512) + 1,512>>>(data, size);
    cudaDeviceSynchronize();
}
