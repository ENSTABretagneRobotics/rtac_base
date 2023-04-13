#ifndef _DEF_RTAC_CUDA_VECTOR_UTILS_H_
#define _DEF_RTAC_CUDA_VECTOR_UTILS_H_

#include <limits>

#include <rtac_base/types/Complex.h>
#include <rtac_base/cuda/CudaVector.h>

namespace rtac { namespace cuda {

float sum(const CudaVector<float>& data, float initial = 0.0f);
float min(const CudaVector<float>& data,
          float initial = std::numeric_limits<float>::max());
float max(const CudaVector<float>& data,
          float initial = std::numeric_limits<float>::min());
float range(const CudaVector<float>& data);

CudaVector<float>& abs(CudaVector<float>& data);
CudaVector<float> sqrt(const CudaVector<float>& data);
CudaVector<float> log(CudaVector<float>& data);

CudaVector<float>&  rescale(CudaVector<float>& data,
                              float minValue = 0.0f, float maxValue = 1.0f);

CudaVector<float>& operator+=(CudaVector<float>& lhs, float a);
CudaVector<float>& operator-=(CudaVector<float>& lhs, float a);
CudaVector<float>& operator*=(CudaVector<float>& lhs, float a);
CudaVector<float>& operator/=(CudaVector<float>& lhs, float a);

CudaVector<float>& operator+=(CudaVector<float>& lhs, const CudaVector<float>& rhs);
CudaVector<float>& operator-=(CudaVector<float>& lhs, const CudaVector<float>& rhs);
CudaVector<float>& operator*=(CudaVector<float>& lhs, const CudaVector<float>& rhs);
CudaVector<float>& operator/=(CudaVector<float>& lhs, const CudaVector<float>& rhs);

CudaVector<float>          abs (const CudaVector<Complex<float>>& data);
CudaVector<float>          real(const CudaVector<Complex<float>>& data);
CudaVector<float>          imag(const CudaVector<Complex<float>>& data);
CudaVector<float>          arg (const CudaVector<Complex<float>>& data);
CudaVector<float>          norm(const CudaVector<Complex<float>>& data);
CudaVector<Complex<float>> conj(const CudaVector<Complex<float>>& data);
CudaVector<Complex<float>> to_complex(const CudaVector<float>& data);

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs, float a);
CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs, float a);
CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs, float a);
CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs, float a);

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs, Complex<float> a);
CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs, Complex<float> a);
CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs, Complex<float> a);
CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs, Complex<float> a);

CudaVector<Complex<float>>& operator+=(CudaVector<Complex<float>>& lhs,
                                         const CudaVector<Complex<float>>& rhs);
CudaVector<Complex<float>>& operator-=(CudaVector<Complex<float>>& lhs,
                                         const CudaVector<Complex<float>>& rhs);
CudaVector<Complex<float>>& operator*=(CudaVector<Complex<float>>& lhs,
                                         const CudaVector<Complex<float>>& rhs);
CudaVector<Complex<float>>& operator/=(CudaVector<Complex<float>>& lhs,
                                         const CudaVector<Complex<float>>& rhs);

CudaVector<float>  abs (const CudaVector<float2>& data);
CudaVector<float>  real(const CudaVector<float2>& data);
CudaVector<float>  imag(const CudaVector<float2>& data);
CudaVector<float>  arg (const CudaVector<float2>& data);
CudaVector<float>  norm(const CudaVector<float2>& data);
CudaVector<float2> conj(const CudaVector<float2>& data);

} // namespace cuda
} // namespace rtac

#endif //_DEF_RTAC_CUDA_VECTOR_UTILS_H_
