#include <iostream>

#include <rtac_base/types/SonarPing.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;

template <typename T>
__global__ void fill_ping(PingView2D<T> ping)
{
    ping(blockIdx.x, threadIdx.x) = ((blockIdx.x + threadIdx.x) & 0x1);
}

int main()
{
    Ping2D<float> p0(Linspace<float>(0.0f, 10.0f, 16),
                     HostVector<float>::linspace(-0.25*3.14, 0.25*3.14, 16));
    Ping2D<float, CudaVector> p1(p0);
    fill_ping<<<p1.height(), p1.width()>>>(p1.view());
    Ping2D<float> p2(p1);

    std::cout << p0 << std::endl;
    std::cout << p1 << std::endl;
    std::cout << p2 << std::endl;

    for(unsigned int h = 0; h < p2.range_count(); h++) { 
        for(unsigned int w = 0; w < p2.bearing_count(); w++) {
            std::cout << ' ' << p2(h,w);
        }
        std::cout << std::endl;
    }

    return 0;
}


