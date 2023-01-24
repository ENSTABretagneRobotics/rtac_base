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
DeviceVector<float>  abs(const DeviceVector<Complex<float>>& data);

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

} // namespace cuda
} // namespace rtac

#endif //_DEF_RTAC_CUDA_VECTOR_UTILS_H_
