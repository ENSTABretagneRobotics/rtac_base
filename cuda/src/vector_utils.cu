#include <rtac_base/cuda/vector_utils.h>

#include <functional>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <cmath>

#include <rtac_base/cuda/vec_math.h>

namespace rtac { namespace cuda {

float sum(const DeviceVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::plus<float>());
}

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

struct thrust_abs// : std::unary_function<float,void>
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

struct thrust_sqrt
{
    __host__ __device__ void operator()(float& x) const { x = ::sqrt(x); }
};


DeviceVector<float> sqrt(const DeviceVector<float>& data)
{
    using namespace thrust::placeholders;

    auto res = data;

    thrust::for_each(thrust::device_pointer_cast(res.data()),
                     thrust::device_pointer_cast(res.data() + data.size()),
                     thrust_sqrt());
    return res;
}

struct thrust_log// : std::unary_function<float,void>
{
    __host__ __device__ void operator()(float& x) const { x = ::log(x); }
};


DeviceVector<float> log(DeviceVector<float>& data)
{
    using namespace thrust::placeholders;

    auto res = data;
    thrust::for_each(thrust::device_pointer_cast(res.data()),
                     thrust::device_pointer_cast(res.data() + data.size()),
                     thrust_log());
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

struct thrust_cabs
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::abs(x);
    }
    __host__ __device__ float operator()(const float2& x) const { 
        return ::sqrt(x.x*x.x + x.y*x.y);
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

struct thrust_creal
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::real(x);
    }
    __host__ __device__ float operator()(const float2& x) const { 
        return x.x;
    }
};

DeviceVector<float> real(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_creal());
    return res;
}

struct thrust_cimag
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::imag(x);
    }
    __host__ __device__ float operator()(const float2& x) const { 
        return x.y;
    }
};

DeviceVector<float> imag(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cimag());
    return res;
}

struct thrust_carg
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::arg(x);
    }
    __host__ __device__ float operator()(const float2& x) const { 
        return ::arg(Complex<float>(x.x, x.y));
    }
};

DeviceVector<float> arg(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_carg());
    return res;
}

struct thrust_cnorm
{
    __host__ __device__ float operator()(const Complex<float>& x) const { 
        return ::norm(x);
    }
    __host__ __device__ float operator()(const float2& x) const { 
        return x.x*x.x + x.y*x.y;
    }
};

DeviceVector<float> norm(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cnorm());
    return res;
}

struct thrust_conj// : std::unary_function<float,void>
{
    __host__ __device__ Complex<float> operator()(const Complex<float>& x) const { 
        return ::conj(x);
    }
    __host__ __device__ float2 operator()(const float2& x) const { 
        return float2{x.x, -x.y};
    }
};

DeviceVector<Complex<float>> conj(const DeviceVector<Complex<float>>& data)
{
    DeviceVector<Complex<float>> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_conj());
    return res;
}

struct thrust_to_complex
{
    __host__ __device__ Complex<float> operator()(const float& x) const { 
        return Complex<float>(x,0.0f);
    }
};

DeviceVector<Complex<float>> to_complex(const DeviceVector<float>& data)
{
    DeviceVector<Complex<float>> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_to_complex());
    return res;
}

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 /= a);
    return lhs;
}

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 /= a);
    return lhs;
}

struct thrust_cplus
{
    __host__ __device__ Complex<float> operator()(const Complex<float>& lhs,
                                                  const Complex<float>& rhs) const
    {
        return lhs + rhs;
    }
    __host__ __device__ float2 operator()(const float2& lhs,
                                          const float2& rhs) const
    {
        return lhs + rhs;
    }
};

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust_cplus());
    return lhs;
}

struct thrust_cminus
{
    __host__ __device__ Complex<float> operator()(const Complex<float>& lhs,
                                                  const Complex<float>& rhs) const
    {
        return lhs - rhs;
    }
    __host__ __device__ float2 operator()(const float2& lhs,
                                          const float2& rhs) const
    {
        return lhs - rhs;
    }
};


DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust_cminus());
    return lhs;
}

struct thrust_cmultiplies
{
    __host__ __device__ Complex<float> operator()(const Complex<float>& lhs,
                                                  const Complex<float>& rhs) const
    {
        return lhs * rhs;
    }
    //__host__ __device__ float2 operator()(const float2& lhs,
    //                                      const float2& rhs) const
    //{
    //    return lhs * rhs;
    //}
};

DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust_cmultiplies());
    return lhs;
}

struct thrust_cdivides
{
    __host__ __device__ Complex<float> operator()(const Complex<float>& lhs,
                                                  const Complex<float>& rhs) const
    {
        return lhs / rhs;
    }
    //__host__ __device__ float2 operator()(const float2& lhs,
    //                                      const float2& rhs) const
    //{
    //    return lhs / rhs;
    //}
};

DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs)
{
    if(lhs.size() != rhs.size()) {
        throw std::runtime_error("Inconsistent vector sizes");
    }

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(lhs.data()),
                      thrust::device_pointer_cast(lhs.data() + lhs.size()),
                      thrust::device_pointer_cast(rhs.data()),
                      thrust::device_pointer_cast(lhs.data()),
                      thrust_cdivides());
    return lhs;
}

DeviceVector<float> abs(const DeviceVector<float2>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cabs());
    return res;
}

DeviceVector<float> real(const DeviceVector<float2>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_creal());
    return res;
}

DeviceVector<float> imag(const DeviceVector<float2>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cimag());
    return res;
}

DeviceVector<float> arg(const DeviceVector<float2>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_carg());
    return res;
}

DeviceVector<float> norm(const DeviceVector<float2>& data)
{
    DeviceVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cnorm());
    return res;
}

DeviceVector<float2> conj(const DeviceVector<float2>& data)
{
    DeviceVector<float2> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_conj());
    return res;
}

} // namespace cuda
} // namespace rtac
