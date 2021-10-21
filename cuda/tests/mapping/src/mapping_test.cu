#include "mapping_test.h"


__global__ void do_map(DeviceMapping<Affine1D,float> mapping, 
                       const float* x, float* out, unsigned int size)
{
    for(int i = 0; i < size; i++) {
        out[i] = mapping(x[i]);
    }
}

DeviceVector<float> map(const DeviceMapping<Affine1D,float>& mapping,
                        const DeviceVector<float>& x)
{
    DeviceVector<float> out(x.size());
    
    do_map<<<1,1>>>(mapping, x.data(), out.data(), out.size());
    cudaDeviceSynchronize();

    return out;
}
