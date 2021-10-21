#ifndef _DEF_RTAC_BASE_CUDA_MAPPING_H_
#define _DEF_RTAC_BASE_CUDA_MAPPING_H_

#include <rtac_base/cuda/Texture2D.h>

namespace rtac { namespace cuda {

/**
 * This struct encodes a generic mapping from an arbitrary input space.  The
 * mapping is encoded into the texture.
 *
 * A Functor type is used to transform the input space input local texture
 * coordinates float2.
 *
 * This can be used to implement fast linear or nearest interpolation on a GPU,
 * colormapping a gray scale image, or simply reading in an image using
 * coordinates in an arbitrary unit.
 *
 * Caution : the result of the operation ax + b must be defined and be
 * implicitly castable to a float2.
 */
template <class FunctorT, typename T>
struct DeviceMapping
{
    cudaTextureObject_t data;
    FunctorT f;

    #ifdef RTAC_CUDACC
    __device__ T operator()(const typename FunctorT::Input& x) const {
        float2 p = f(x);
        return tex2D<T>(data, p.x, p.y);
    }
    #endif //RTAC_CUDACC
};

}; //namespace cuda
}; //namespace rtac

#ifndef RTAC_CUDACC

namespace rtac { namespace cuda {

/**
 * Host side class with manage device side Device mapping data.
 */
template <class FunctorT, typename T>
class Mapping
{
    public:

    using Ptr       = std::shared_ptr<Mapping>;
    using ConstPtr  = std::shared_ptr<const Mapping>;
    using Texture   = Texture2D<T>;
    using DeviceMap = DeviceMapping<FunctorT,T>;

    protected:
    
    Texture  data_;
    FunctorT f_;
    
    Mapping(const FunctorT& f, Texture&& data);

    public:

    static Ptr Create(const FunctorT& f, Texture&& data);
    static Ptr Create(const FunctorT& f, const Texture& data);

    void set_texture(const Texture& texture);
    void set_texture(Texture&& texture);
    void set_FunctorT(const FunctorT& a);

    const Texture& texture() const;
    const FunctorT& functor() const;

    DeviceMap device_map() const;
};

template <class FunctorT, typename T>
Mapping<FunctorT,T>::Mapping(const FunctorT& f, Texture&& data) :
    data_(data),
    f_(f)
{}

template <class FunctorT, typename T>
typename Mapping<FunctorT,T>::Ptr 
Mapping<FunctorT,T>::Create(const FunctorT& f, const Texture& data)
{
    return Ptr(new Mapping<FunctorT,T>(f, data));
}

template <class FunctorT, typename T>
typename Mapping<FunctorT,T>::Ptr 
Mapping<FunctorT,T>::Create(const FunctorT& f, Texture&& data)
{
    return Ptr(new Mapping<FunctorT,T>(f, std::move(Texture(data))));
}

template <class FunctorT, typename T>
void Mapping<FunctorT,T>::set_texture(const Texture& texture)
{
    data_ = texture;
}

template <class FunctorT, typename T>
void Mapping<FunctorT,T>::set_texture(Texture&& texture)
{
    data_ = texture;
}

template <class FunctorT, typename T>
void Mapping<FunctorT,T>::set_FunctorT(const FunctorT& f)
{
    f_ = f;
}

template <class FunctorT, typename T>
const Texture2D<T>& Mapping<FunctorT,T>::texture() const
{
    return data_;
}

template <class FunctorT, typename T>
const FunctorT& Mapping<FunctorT,T>::functor() const
{
    return f_;
}

template <class FunctorT, typename T>
typename Mapping<FunctorT,T>::DeviceMap Mapping<FunctorT,T>::device_map() const
{
    return DeviceMapping<FunctorT,T>({data_.texture(), f_});
}

}; //namespace cuda
}; //namespace rtac

#endif //RTAC_CUDACC

#endif //_DEF_RTAC_BASE_CUDA_MAPPING_H_
