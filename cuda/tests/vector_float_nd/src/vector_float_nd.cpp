#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/time.h>
using namespace rtac::time;

#include <rtac_base/cuda/VectorFloatND.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include "vector_float_nd.h"

int main()
{
    VectorFloatND<float, std::vector> v1(10);
    for(int i = 0; i < v1.size(); i++) {
        v1.set(i,i);
    }

    VectorFloatND<float, DeviceVector> d1(v1);
    VectorFloatND<float, HostVector> h1(d1);
    for(int i = 0; i < h1.size(); i++) {
        cout << " " << h1[i];
    }
    cout << endl;

    VectorFloatND<float3, DeviceVector> d3(10);
    fill(d3.view());
    cudaDeviceSynchronize();
    VectorFloatND<float3, HostVector> h3(d3);
    for(int i = 0; i < h3.size(); i++) {
        cout << h3[i].x << " " << h3[i].y << " " << h3[i].z << endl;
    }

    //speed test

    using TestT = float3;
    //using TestT = float4;
    unsigned int blockSize = 1024;
    unsigned int I = 1000;
    unsigned int N = 20000000;
    VectorFloatND<TestT, DeviceVector> dTest0(N);
    VectorFloatND<TestT, HostVector>   hTest0(N);
    Clock clock;
    
    clock.reset();
    for(int i = 0; i < I; i++) {
        update(dTest0.view());
        //hTest0 = dTest0;
    }
    auto dt0 = clock.interval();

    DeviceVector<TestT> dTest1(N);
    HostVector<TestT>   hTest1(N);

    clock.reset();
    for(int i = 0; i < I; i++) {
        update(dTest1.view());
        //hTest1 = dTest1;
    }
    auto dt1 = clock.interval();

    std::cout << "Ellapsed 0 : " << 1000.0*dt0 / I << "ms" << std::endl;
    std::cout << "Ellapsed 1 : " << 1000.0*dt1 / I << "ms" << std::endl;
    std::cout << "sizeof(float3) : " << sizeof(float3) << std::endl;


    return 0;
}
