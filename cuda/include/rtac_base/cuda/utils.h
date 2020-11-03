#ifndef _DEF_RTAC_BASE_CUDA_UTILS_H_
#define _DEF_RTAC_BASE_CUDA_UTILS_H_

#include <iostream>
#include <sstream>

namespace rtac { namespace cuda {

// inline won't link properly
void check_error(unsigned int code);

// inline won't link properly
unsigned int do_malloc(void** devPtr, size_t size);
// inline won't link properly
unsigned int do_free(void* devPtr);

template <typename T>
inline T* alloc(size_t size)
{
    T* res;
    check_error(do_malloc(reinterpret_cast<void**>(&res), size*sizeof(T)));
    return res;
}

template <typename T>
inline void free(T* devPtr)
{
    check_error(do_free(reinterpret_cast<void*>(devPtr)));
}

struct memcpy
{
    static unsigned int copy_device_to_host(void* dst, const void* src, size_t count);
    static unsigned int copy_host_to_device(void* dst, const void* src, size_t count);
    static unsigned int copy_device_to_device(void* dst, const void* src, size_t count);
    static unsigned int copy_host_to_host    (void* dst, const void* src, size_t count);

    enum Kind { HostToHost, HostToDevice, DeviceToHost, DeviceToDevice };
    static unsigned int do_copy(void* dst, const void* src, size_t count, Kind kind);

    template <typename T>
    static void device_to_host(T* dst, const T* src, size_t count)
    {
        check_error(copy_device_to_host(reinterpret_cast<void*>(dst),
                                        reinterpret_cast<const void*>(src),
                                        sizeof(T)*count));
    }

    template <typename T>
    static void host_to_device(T* dst, const T* src, size_t count)
    {
        check_error(copy_host_to_device(reinterpret_cast<void*>(dst),
                                        reinterpret_cast<const void*>(src),
                                        sizeof(T)*count));
    }

    template <typename T>
    static void device_to_device(T* dst, const T* src, size_t count)
    {
        check_error(copy_device_to_device(reinterpret_cast<void*>(dst),
                                          reinterpret_cast<const void*>(src),
                                          sizeof(T)*count));
    }

    template <typename T>
    static void host_to_host(T* dst, const T* src, size_t count)
    {
        check_error(copy_host_to_host(reinterpret_cast<void*>(dst),
                                      reinterpret_cast<const void*>(src),
                                      sizeof(T)*count));
    }
};

}; //namespace cuda
}; //namespace rtac


#endif //_DEF_RTAC_BASE_CUDA_UTILS_H_
