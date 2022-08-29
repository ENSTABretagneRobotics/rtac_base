#ifndef _DEF_RTAC_CUDA_VECTOR_FLOAT_ND_H_
#define _DEF_RTAC_CUDA_VECTOR_FLOAT_ND_H_

#include <tuple>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/types/VectorView.h>

namespace rtac { namespace cuda {

template <typename T,
          template<typename>class VectorT>
struct CudaFloatInfo {
    // This assert fails only if CudaFloatInfo is not instanciated properly
    static_assert(sizeof(T) != 0,
        "Type for struct info should be either float,float2,float3,float4");
};
template<template<typename>class VectorT> struct CudaFloatInfo<float, VectorT> {
    using Scalar = float;
    static constexpr unsigned int Dimension = 1;
    using ContainerType = std::tuple<VectorT<float>>;
};
template<template<typename>class VectorT> struct CudaFloatInfo<float2, VectorT> {
    using Scalar = float;
    static constexpr unsigned int Dimension = 2;
    using ContainerType = std::tuple<VectorT<float>,VectorT<float>>;
};
template<template<typename>class VectorT> struct CudaFloatInfo<float3, VectorT> {
    using Scalar = float;
    static constexpr unsigned int Dimension = 3;
    using ContainerType = std::tuple<VectorT<float>,VectorT<float>,VectorT<float>>;
};
template<template<typename>class VectorT> struct CudaFloatInfo<float4, VectorT> {
    using Scalar = float;
    static constexpr unsigned int Dimension = 4;
    using ContainerType = std::tuple<VectorT<float>,VectorT<float>,VectorT<float>,VectorT<float>>;
};

template <typename T,
          template<typename>class VectorT>
struct ContainerType {};

template <template<typename>class VectorT>
struct ContainerType<float,VectorT> {
    using type = std::tuple<VectorT<float>>;
};
template <template<typename>class VectorT>
struct ContainerType<float2,VectorT> {
    using type = std::tuple<VectorT<float>,VectorT<float>>;
};
template <template<typename>class VectorT>
struct ContainerType<float3,VectorT> {
    using type = std::tuple<VectorT<float>,VectorT<float>,VectorT<float>>;
};
template <template<typename>class VectorT>
struct ContainerType<float4,VectorT> {
    using type = std::tuple<VectorT<float>,VectorT<float>,VectorT<float>,VectorT<float>>;
};

template <typename T,
          template<typename>class VectorT>
class VectorFloatND
{
    public:

    using value_type = T;
    using ScalarType = typename CudaFloatInfo<T,VectorT>::Scalar;
    // This is a std::tuple of VectorT<T>. Size of tuple is Dimension
    //using Container = typename ContainerType<T,VectorT>::type;
    using Container = typename CudaFloatInfo<T,VectorT>::ContainerType;
    static constexpr unsigned int Dimension  = CudaFloatInfo<T,VectorT>::Dimension;

    protected:

    Container data_;

    public:

    VectorFloatND() {}
    VectorFloatND(std::size_t size) { this->resize(size); }
    VectorFloatND(const Container& data) : data_(data) {}
    template<template<typename>class VectorT2>
    VectorFloatND(const VectorFloatND<T,VectorT2>& other) : data_(other.container()) {}

    template<template<typename>class VectorT2>
    VectorFloatND<T,VectorT>& operator=(const VectorFloatND<T,VectorT2>& other) {
        data_ = other.container();
        return *this;
    }

    const Container& container() const { return data_; }
    Container&       container()       { return data_; }
    
    void resize(size_t size);
    RTAC_HOSTDEVICE size_t size() const;
    RTAC_HOSTDEVICE size_t capacity() const;

    RTAC_HOSTDEVICE T operator[](std::size_t idx) const;
    RTAC_HOSTDEVICE void set(std::size_t idx, T value);

    RTAC_HOSTDEVICE T front() const { return this->operator[](0); }
    RTAC_HOSTDEVICE T back()  const { return this->operator[](this->size() - 1); }
    RTAC_HOSTDEVICE void set_front(T value) { this->set(0, value); }
    RTAC_HOSTDEVICE void set_back(T value)  { this->set(this->size() - 1, value); }

    VectorFloatND<const T,rtac::types::VectorView> view() const;
    VectorFloatND<T,rtac::types::VectorView> view();
};

template <typename T>
using VectorFloatNDView = VectorFloatND<T, rtac::types::VectorView>;

template <typename T, template<typename>class V>
inline void VectorFloatND<T,V>::resize(size_t size)
{
    std::get<0>(data_).resize(size);
    if constexpr(Dimension > 1) std::get<1>(data_).resize(size);
    if constexpr(Dimension > 2) std::get<2>(data_).resize(size);
    if constexpr(Dimension > 3) std::get<3>(data_).resize(size);
}

#pragma hd_warning_disable
template <typename T, template<typename>class V>
RTAC_HOSTDEVICE inline size_t VectorFloatND<T,V>::size() const
{
    return std::get<0>(data_).size();
}

#pragma hd_warning_disable
template <typename T, template<typename>class V>
RTAC_HOSTDEVICE inline size_t VectorFloatND<T,V>::capacity() const
{
    return std::get<0>(data_).capacity();
}

#pragma hd_warning_disable
template <typename T, template<typename>class V>
RTAC_HOSTDEVICE inline T VectorFloatND<T,V>::operator[](size_t idx) const
{
    if constexpr(Dimension == 1) {
        return std::get<0>(data_)[idx];
    }
    if constexpr(Dimension == 2) {
        return T{std::get<0>(data_)[idx],
                 std::get<1>(data_)[idx]};
    }
    if constexpr(Dimension == 3) {
        return T{std::get<0>(data_)[idx],
                 std::get<1>(data_)[idx],
                 std::get<2>(data_)[idx]};
    }
    if constexpr(Dimension == 4) {
        return T{std::get<0>(data_)[idx],
                 std::get<1>(data_)[idx],
                 std::get<2>(data_)[idx],
                 std::get<3>(data_)[idx]};
    }
}

#pragma hd_warning_disable
template <typename T, template<typename>class V>
RTAC_HOSTDEVICE inline void VectorFloatND<T,V>::set(std::size_t idx, T value)
{
    if constexpr(Dimension == 1) std::get<0>(data_)[idx] = value;
    if constexpr(Dimension > 1) {
        std::get<0>(data_)[idx] = value.x;
        std::get<1>(data_)[idx] = value.y;
    }
    if constexpr(Dimension > 2) std::get<2>(data_)[idx] = value.z;
    if constexpr(Dimension > 3) std::get<3>(data_)[idx] = value.w;
}

template <typename T, template<typename>class V>
VectorFloatND<const T,rtac::types::VectorView> VectorFloatND<T,V>::view() const
{
    if constexpr(Dimension == 1) {
        return VectorFloatND<const T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view()));
    }
    if constexpr(Dimension == 2) {
        return VectorFloatND<const T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view()));
    }
    if constexpr(Dimension == 3) {
        return VectorFloatND<const T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view(),
                            std::get<2>(data_).view()));
    }
    if constexpr(Dimension == 4) {
        return VectorFloatND<const T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view(),
                            std::get<2>(data_).view(),
                            std::get<3>(data_).view()));
    }
}

template <typename T, template<typename>class V>
VectorFloatND<T,rtac::types::VectorView> VectorFloatND<T,V>::view()
{
    if constexpr(Dimension == 1) {
        return VectorFloatND<T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view()));
    }
    if constexpr(Dimension == 2) {
        return VectorFloatND<T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view()));
    }
    if constexpr(Dimension == 3) {
        return VectorFloatND<T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view(),
                            std::get<2>(data_).view()));
    }
    if constexpr(Dimension == 4) {
        return VectorFloatND<T,rtac::types::VectorView>(
            std::make_tuple(std::get<0>(data_).view(),
                            std::get<1>(data_).view(),
                            std::get<2>(data_).view(),
                            std::get<3>(data_).view()));
    }
}

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_VECTOR_FLOAT_ND_H_
