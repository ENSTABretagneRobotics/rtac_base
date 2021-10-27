#ifndef _DEF_RTAC_BASE_CUDA_MAPPING_H_
#define _DEF_RTAC_BASE_CUDA_MAPPING_H_

#include <rtac_base/cuda/Texture2D.h>
#include <rtac_base/cuda/FunctorCompound.h>
#include <rtac_base/cuda/functors.h>

namespace rtac { namespace cuda {

/**
 * Mappings are special classes of functors which output value is fetched from
 * a texture. The main purpose of this type is to leverage the fast local
 * caching and interpolation of GPU texture units. This is especially relevant
 * for operations such as correlations or convolutions.
 *
 * Mapping are handled differently than regular functors because the texture
 * memory must be managed from host side (creation, memory allocation and
 * destruction...). On the device side however, a mapping behaves like a
 * regular functor.
 */
template <typename T>
struct DeviceMapping2D
{
    // Declaring input and output types to be compatible with other functors.
    using InputT  = float2;
    using OutputT = T;

    cudaTextureObject_t data;

    #ifdef RTAC_CUDACC // this cannot be used from host side because of the texture fetch
    __device__ T operator()(const float2& uv) const {
        return tex2D<T>(data, uv.x, uv.y);
    }
    #endif //RTAC_CUDACC
};

template <typename T>
struct DeviceMapping1D
{
    // Declaring input and output types to be compatible with other functors.
    using InputT  = float;
    using OutputT = T;

    cudaTextureObject_t data;

    #ifdef RTAC_CUDACC // this cannot be used from host side because of the texture fetch
    __device__ T operator()(float u) const {
        return tex2D<T>(data, u, 0.0f);
    }
    #endif //RTAC_CUDACC
};

/**
 * This is a convenience definition allowing to deduce a full DeviceMap type
 * depending on the given FunctorT. Is used primarily in the Mapping type.
 */
template <typename T, class FunctorT>
struct device_map_type {
    using FoutT = typename FunctorT::OutputT;

    static_assert(std::is_same<float, FoutT>::value || std::is_same<float2, FoutT>::value,
                  "For a cuda::Mapping type, the output type of the optional functor must be either float or float2");
    
    // Deducing DeviceMap type to use (1D or 2D), depending on the
    // FunctorT::OutputT type.
    using BaseDeviceMapT = typename std::conditional<std::is_same<float, FoutT>::value,
                                                     DeviceMapping1D<T>,
                                                     DeviceMapping2D<T>>::type;
    
    // Building final type.
    using type = functors::FunctorCompound<BaseDeviceMapT, FunctorT>;
};

/**
 * Host side class with manage device side Device mapping data. This also
 * handle an additional functor which allows to perform an operation on the
 * input coordinates (namely a nomalization between 0 and 1) before effectively
 * performing a texture fetch.
 */
template <typename T, class FunctorT = functors::IdentityFunctor<float2>>
class Mapping
{
    public:

    using Ptr       = std::shared_ptr<Mapping>;
    using ConstPtr  = std::shared_ptr<const Mapping>;
    using Texture   = Texture2D<T>;
    using BaseDeviceMapT = typename device_map_type<T, FunctorT>::BaseDeviceMapT;
    using DeviceMap      = typename device_map_type<T, FunctorT>::type;

    protected:
    
    Texture  data_;
    FunctorT f_;
    
    Mapping(Texture&& data);
    Mapping(Texture&& data, const FunctorT& f);

    public:

    static Ptr Create(Texture&& data);
    static Ptr Create(const Texture& data);

    static Ptr Create(Texture&& data, const FunctorT& f);
    static Ptr Create(const Texture& data, const FunctorT& f);

    void set_texture(const Texture& texture);
    void set_texture(Texture&& texture);
    void set_FunctorT(const FunctorT& f);

    const Texture& texture() const;
    const FunctorT& functor() const;

    DeviceMap device_map() const;
};

template <typename T, class FunctorT>
Mapping<T,FunctorT>::Mapping(Texture&& data) :
    data_(data)
{}

template <typename T, class FunctorT>
Mapping<T,FunctorT>::Mapping(Texture&& data, const FunctorT& f) :
    data_(data),
    f_(f)
{}

template <typename T, class FunctorT>
typename Mapping<T,FunctorT>::Ptr 
Mapping<T,FunctorT>::Create(Texture&& data)
{
    return Ptr(new Mapping<T,FunctorT>(std::move(Texture(data))));
}

template <typename T, class FunctorT>
typename Mapping<T,FunctorT>::Ptr 
Mapping<T,FunctorT>::Create(const Texture& data)
{
    return Ptr(new Mapping<T,FunctorT>(std::move(Texture(data))));
}

template <typename T, class FunctorT>
typename Mapping<T,FunctorT>::Ptr 
Mapping<T,FunctorT>::Create(Texture&& data, const FunctorT& f)
{
    return Ptr(new Mapping<T,FunctorT>(std::move(Texture(data)), f));
}

template <typename T, class FunctorT>
typename Mapping<T,FunctorT>::Ptr 
Mapping<T,FunctorT>::Create(const Texture& data, const FunctorT& f)
{
    return Ptr(new Mapping<T,FunctorT>(std::move(Texture(data)), f));
}

template <typename T, class FunctorT>
void Mapping<T,FunctorT>::set_texture(const Texture& texture)
{
    data_ = texture;
}

template <typename T, class FunctorT>
void Mapping<T,FunctorT>::set_texture(Texture&& texture)
{
    data_ = texture;
}

template <typename T, class FunctorT>
void Mapping<T,FunctorT>::set_FunctorT(const FunctorT& f)
{
    f_ = f;
}

template <typename T, class FunctorT>
const Texture2D<T>& Mapping<T,FunctorT>::texture() const
{
    return data_;
}

template <typename T, class FunctorT>
const FunctorT& Mapping<T,FunctorT>::functor() const
{
    return f_;
}

template <typename T, class FunctorT>
typename Mapping<T,FunctorT>::DeviceMap Mapping<T,FunctorT>::device_map() const
{
    return DeviceMap(BaseDeviceMapT({data_.texture()}), f_);
}

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_MAPPING_H_
