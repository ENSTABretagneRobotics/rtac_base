#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda {

//inline
void check_error(unsigned int code)
{
    if(code != cudaSuccess) {
        std::ostringstream oss;
        oss << "CUDA error " << code << " : " 
            << cudaGetErrorString(static_cast<cudaError_t>(code));
        throw std::runtime_error(oss.str());
    }
}

void set_device(int deviceOrdinal)
{
    check_error(cudaSetDevice(deviceOrdinal));
}

//inline
unsigned int do_malloc(void** devPtr, size_t size)
{
    return cudaMalloc(devPtr, size);
}

//inline
unsigned int do_free(void* devPtr)
{
    return cudaFree(devPtr);
}

unsigned int memcpy::copy_device_to_host(void* dst, const void* src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

unsigned int memcpy::copy_host_to_device(void* dst, const void* src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

unsigned int memcpy::copy_device_to_device(void* dst, const void* src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

unsigned int memcpy::copy_host_to_host(void* dst, const void* src, size_t count)
{
    return cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
}

unsigned int memcpy::do_copy(void* dst, const void* src, size_t count, Kind kind)
{
    switch(kind) {
        case HostToHost:
            return memcpy::copy_host_to_host(dst, src, count);
        case HostToDevice:
            return memcpy::copy_host_to_device(dst, src, count);
        case DeviceToHost:
            return memcpy::copy_device_to_host(dst, src, count);
        case DeviceToDevice:
            return memcpy::copy_device_to_device(dst, src, count);
        default:
            std::ostringstream oss;
            oss << "rtac::cuda::memcpy::copy : "
                << "invalid memory transfert direction parameter";
            throw std::runtime_error(oss.str());
    }
}


}; //namespace cuda
}; //namespace rtac
