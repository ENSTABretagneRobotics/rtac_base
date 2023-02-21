#ifndef _DEF_RTAC_CUDA_VECTOR_UTILS_H_
#define _DEF_RTAC_CUDA_VECTOR_UTILS_H_

#include <limits>

#include <rtac_base/types/Complex.h>
#include <rtac_base/cuda/DeviceVector.h>

namespace rtac { namespace cuda {

float min(const DeviceVector<float>& data,
          float initial = std::numeric_limits<float>::max());
float max(const DeviceVector<float>& data,
          float initial = std::numeric_limits<float>::min());
float range(const DeviceVector<float>& data);

DeviceVector<float>& abs(DeviceVector<float>& data);
DeviceVector<float> sqrt(const DeviceVector<float>& data);

DeviceVector<float>&  rescale(DeviceVector<float>& data,
                              float minValue = 0.0f, float maxValue = 1.0f);

DeviceVector<float>& operator+=(DeviceVector<float>& lhs, float a);
DeviceVector<float>& operator-=(DeviceVector<float>& lhs, float a);
DeviceVector<float>& operator*=(DeviceVector<float>& lhs, float a);
DeviceVector<float>& operator/=(DeviceVector<float>& lhs, float a);

DeviceVector<float>& operator+=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs);
DeviceVector<float>& operator-=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs);
DeviceVector<float>& operator*=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs);
DeviceVector<float>& operator/=(DeviceVector<float>& lhs, const DeviceVector<float>& rhs);

DeviceVector<float>          abs (const DeviceVector<Complex<float>>& data);
DeviceVector<float>          real(const DeviceVector<Complex<float>>& data);
DeviceVector<float>          imag(const DeviceVector<Complex<float>>& data);
DeviceVector<float>          arg (const DeviceVector<Complex<float>>& data);
DeviceVector<float>          norm(const DeviceVector<Complex<float>>& data);
DeviceVector<Complex<float>> conj(const DeviceVector<Complex<float>>& data);
DeviceVector<Complex<float>> to_complex(const DeviceVector<float>& data);

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs, float a);
DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs, float a);
DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs, float a);
DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs, float a);

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs, Complex<float> a);
DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs, Complex<float> a);
DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs, Complex<float> a);
DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs, Complex<float> a);

DeviceVector<Complex<float>>& operator+=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs);
DeviceVector<Complex<float>>& operator-=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs);
DeviceVector<Complex<float>>& operator*=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs);
DeviceVector<Complex<float>>& operator/=(DeviceVector<Complex<float>>& lhs,
                                         const DeviceVector<Complex<float>>& rhs);

DeviceVector<float>  abs (const DeviceVector<float2>& data);
DeviceVector<float>  real(const DeviceVector<float2>& data);
DeviceVector<float>  imag(const DeviceVector<float2>& data);
DeviceVector<float>  arg (const DeviceVector<float2>& data);
DeviceVector<float>  norm(const DeviceVector<float2>& data);
DeviceVector<float2> conj(const DeviceVector<float2>& data);

} // namespace cuda
} // namespace rtac

#endif //_DEF_RTAC_CUDA_VECTOR_UTILS_H_
