#include "functors_test.h"
#include "functors_test.hcu"

namespace rtac { namespace cuda {

DeviceVector<float> scaling(const DeviceVector<float>& input,
                            const functors::Scaling<float>& func)
{
    DeviceVector<float> output(input.size());

    apply_functor<<<1,1>>>(output.data(), input.data(), func, input.size());
    cudaDeviceSynchronize();

    return output;
}

DeviceVector<float> saxpy(const DeviceVector<float>& input, const Saxpy& func)
{
    DeviceVector<float> output(input.size());

    apply_functor<<<1,1>>>(output.data(), input.data(), func, input.size());
    cudaDeviceSynchronize();

    return output;
}

}; //namespace cuda
}; //namespace rtac
