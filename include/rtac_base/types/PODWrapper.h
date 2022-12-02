#ifndef _DEF_RTAC_BASE_CUDA_CONSTANT_OBJECT_H_
#define _DEF_RTAC_BASE_CUDA_CONSTANT_OBJECT_H_

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace cuda {

/**
 * The purpose of this object is hide the details of a regular c++ class type
 * into a Plain Old Data type. The primairy purpose of this type is to be able
 * to use non-trivially constructible objects in CUDA __constant__ memory.
 *
 * This type is to be used like a pointer to a Derived object.
 */

template <class Derived>
struct PODWrapper 
{
    uint8_t data_[sizeof(Derived)];

    RTAC_HOSTDEVICE const Derived& operator*() const { 
        return *reinterpret_cast<const Derived*>(&data_[0]);
    }
    RTAC_HOSTDEVICE Derived& operator*() { 
        return *reinterpret_cast<Derived*>(&data_[0]);
    }

    RTAC_HOSTDEVICE const Derived* operator->() const { 
        return reinterpret_cast<const Derived*>(&data_[0]);
    }
    RTAC_HOSTDEVICE Derived* operator->() { 
        return reinterpret_cast<Derived*>(&data_[0]);
    }
};

} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_CONSTANT_OBJECT_H_
