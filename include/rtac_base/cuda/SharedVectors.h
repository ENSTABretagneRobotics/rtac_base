#ifndef _DEF_RTAC_BASE_CUDA_SHARED_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_SHARED_VECTOR_H_

#include <iostream>

#include <rtac_base/types/SharedVector.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace rtac { namespace cuda {

// Stub only there to make thrust::device_vector return a raw pointer on data().
template <typename T>
class DeviceVectorAdapter : public thrust::device_vector<T>
{
    public:

    using value_type = T;
    
    // universal constructor (all arguments are passed to
    // thrust::device_vector<T> constructor). Check if no problem with this,
    // might be a bit over the top.
    template <class... Args>
    DeviceVectorAdapter(Args... args) : thrust::device_vector<T>(args...) {}

    template <typename VectorT>
    DeviceVectorAdapter<T>& operator=(const VectorT& other)
    {
        this->thrust::device_vector<T>::operator=(other);
        return *this;
    }

    value_type* data()
    {
        return thrust::raw_pointer_cast(this->thrust::device_vector<T>::data());
    }

    const value_type* data() const
    {
        return thrust::raw_pointer_cast(this->thrust::device_vector<T>::data());
    }
};

template <typename T>
using DeviceVector = rtac::types::SharedVectorBase<DeviceVectorAdapter<T>>;
template <typename T>
using HostVector = rtac::types::SharedVectorBase<thrust::host_vector<T>>;

}; //namespace cuda
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::DeviceVector<T>& v)
{
    os << rtac::cuda::HostVector<T>(v);
    return os;
}

#endif //_DEF_RTAC_BASE_CUDA_SHARED_VECTOR_H_
