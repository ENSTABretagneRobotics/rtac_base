#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/TextureVector.h>
using namespace rtac;
using namespace rtac::cuda;

template <typename T>
__global__ void print(TextureVectorView<T> data)
{
    for(int i = 0; i < data.size(); i++) {
        printf(" %f", data[i]);
    }
    printf("\n");
}

int main()
{
    TextureVector<float> data(HostVector<float>::linspace(0.0,1.0,16));

    print<<<1,1>>>(data.view());
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    return 0;
}
