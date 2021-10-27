#ifndef _DEF_RTAC_BASE_CUDA_MAPPINGS_H_
#define _DEF_RTAC_BASE_CUDA_MAPPINGS_H_

#include <rtac_base/cuda/functors.h>
#include <rtac_base/cuda/Mapping.h>

namespace rtac { namespace cuda {

template <typename T>
struct Mapping1D : public Mapping<T, functors::AffineTransform<float>>
{
    using MappingT   = Mapping<T, functors::AffineTransform<float>>;
    using Ptr        = typename MappingT::Ptr;
    using ConstPtr   = typename MappingT::ConstPtr;
    using DeviceMap  = typename MappingT::DeviceMap;
    
    static Ptr Create(float domainMin, float domainMax,
                      const T* domainData, unsigned int dataSize)
    {
        Texture2D<T> tex;
        tex.set_image(dataSize, 1, domainData);

        auto t = functors::AffineTransform<float>(
            {1.0f / (domainMax - domainMin), 
            -domainMin / (domainMax - domainMin)});

        return MappingT::Create(std::move(tex), t);
    }
};

}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_MAPPINGS_H_
