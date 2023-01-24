#include <rtac_base/cuda/vector_utils.h>

#include <functional>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <cmath>

namespace rtac { namespace cuda {

float min(const DeviceVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::minimum<float>());
}

float max(const DeviceVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::maximum<float>());
}

float range(const DeviceVector<float>& data) 
{
    return max(data) - min(data);
}

struct thrust_abs : std::unary_function<float,void>
{
    __host__ __device__ void operator()(float& x) const { x = fabs(x); }
};


DeviceVector<float>& abs(DeviceVector<float>& data)
{
    using namespace thrust::placeholders;

    thrust::for_each(thrust::device_pointer_cast(data.data()),
                     thrust::device_pointer_cast(data.data() + data.size()),
                     thrust_abs());
    return data;
}

struct thrust_cabs : std::unary_function<float,void>
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::abs(x);
    }
};

DeviceVector<float> abs(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cabs());
    return res;
}

DeviceVector<float>& rescale(DeviceVector<float>& data, float minValue, float maxValue)
{
    using namespace thrust::placeholders;

    float dataMin = min(data);
    float dataMax = max(data);
    
    float a = (maxValue - minValue) / (dataMax - dataMin);
    float b = -a * dataMin + minValue;
    thrust::for_each(thrust::device_pointer_cast(data.data()),
                     thrust::device_pointer_cast(data.data() + data.size()),
                     _1 = a * _1 + b);
    return data;
}

DeviceVector<float>& operator+=(DeviceVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

DeviceVector<float>& operator-=(DeviceVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

DeviceVector<float>& operator*=(DeviceVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

DeviceVector<float>& operator/=(DeviceVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 /= a);
    return lhs;
}

DeviceVector<float>& operator+=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::plus<float>());
    return lhs;
}

DeviceVector<float>& operator-=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::minus<float>());
    return lhs;
}

DeviceVector<float>& operator*=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::multiplies<float>());
    return lhs;
}

DeviceVector<float>& operator/=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::divides<float>());
    return lhs;
}

} // namespace cuda
} // namespace rtac
