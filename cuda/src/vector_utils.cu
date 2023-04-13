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

float sum(const CudaVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::plus<float>());
}

float min(const CudaVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::minimum<float>());
}

float max(const CudaVector<float>& data, float initial)
{
    return thrust::reduce(thrust::device_pointer_cast(data.data()),
                          thrust::device_pointer_cast(data.data() + data.size()),
                          initial, thrust::maximum<float>());
}

float range(const CudaVector<float>& data) 
{
    return max(data) - min(data);
}

struct thrust_abs// : std::unary_function<float,void>
{
    __host__ __device__ void operator()(float& x) const { x = fabs(x); }
};


CudaVector<float>& abs(CudaVector<float>& data)
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


CudaVector<float> sqrt(const CudaVector<float>& data)
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


CudaVector<float> log(CudaVector<float>& data)
{
    using namespace thrust::placeholders;

    auto res = data;
    thrust::for_each(thrust::device_pointer_cast(res.data()),
                     thrust::device_pointer_cast(res.data() + data.size()),
                     thrust_log());
    return res;
}


CudaVector<float>& rescale(CudaVector<float>& data, float minValue, float maxValue)
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

CudaVector<float>& operator+=(CudaVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

CudaVector<float>& operator-=(CudaVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

CudaVector<float>& operator*=(CudaVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

CudaVector<float>& operator/=(CudaVector<float>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 /= a);
    return lhs;
}

CudaVector<float>& operator+=(CudaVector<float>& lhs, const CudaVector<float>& rhs)
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

CudaVector<float>& operator-=(CudaVector<float>& lhs, const CudaVector<float>& rhs)
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

CudaVector<float>& operator*=(CudaVector<float>& lhs, const CudaVector<float>& rhs)
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

CudaVector<float>& operator/=(CudaVector<float>& lhs, const CudaVector<float>& rhs)
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

CudaVector<float> abs(const CudaVector<Complex<float>>& data)
{
    CudaVector<float> res(data.size());

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

CudaVector<float> real(const CudaVector<Complex<float>>& data)
{
    CudaVector<float> res(data.size());

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

CudaVector<float> imag(const CudaVector<Complex<float>>& data)
{
    CudaVector<float> res(data.size());

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

CudaVector<float> arg(const CudaVector<Complex<float>>& data)
{
    CudaVector<float> res(data.size());

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

CudaVector<float> norm(const CudaVector<Complex<float>>& data)
{
    CudaVector<float> res(data.size());

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

CudaVector<Complex<float>> conj(const CudaVector<Complex<float>>& data)
{
    CudaVector<Complex<float>> res(data.size());

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

CudaVector<Complex<float>> to_complex(const CudaVector<float>& data)
{
    CudaVector<Complex<float>> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_to_complex());
    return res;
}

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs, float a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 /= a);
    return lhs;
}

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 += a);
    return lhs;
}

CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 -= a);
    return lhs;
}

CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs, Complex<float> a)
{
    using namespace thrust::placeholders;
    thrust::for_each(thrust::device,
                     thrust::device_pointer_cast(lhs.data()),
                     thrust::device_pointer_cast(lhs.data() + lhs.size()),
                     _1 *= a);
    return lhs;
}

CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs, Complex<float> a)
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

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs,
                                       const CudaVector<Complex<float>>& rhs)
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


CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs,
                                       const CudaVector<Complex<float>>& rhs)
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

CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs,
                                       const CudaVector<Complex<float>>& rhs)
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

CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs,
                                       const CudaVector<Complex<float>>& rhs)
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

CudaVector<float> abs(const CudaVector<float2>& data)
{
    CudaVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cabs());
    return res;
}

CudaVector<float> real(const CudaVector<float2>& data)
{
    CudaVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_creal());
    return res;
}

CudaVector<float> imag(const CudaVector<float2>& data)
{
    CudaVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cimag());
    return res;
}

CudaVector<float> arg(const CudaVector<float2>& data)
{
    CudaVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_carg());
    return res;
}

CudaVector<float> norm(const CudaVector<float2>& data)
{
    CudaVector<float> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_cnorm());
    return res;
}

CudaVector<float2> conj(const CudaVector<float2>& data)
{
    CudaVector<float2> res(data.size());

    thrust::transform(thrust::device,
                      thrust::device_pointer_cast(data.data()),
                      thrust::device_pointer_cast(data.data() + data.size()),
                      thrust::device_pointer_cast(res.data()),
                      thrust_conj());
    return res;
}

} // namespace cuda
} // namespace rtac
