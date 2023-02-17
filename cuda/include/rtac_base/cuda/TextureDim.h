#ifndef _DEF_RTAC_BASE_CUDA_TEXTURE_DIM_H_
#define _DEF_RTAC_BASE_CUDA_TEXTURE_DIM_H_

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/containers/DimExpression.h>
#include <rtac_base/cuda/TextureVector.h>

namespace rtac { namespace cuda {

struct TextureDimView : public DimExpression<TextureDimView>
{
    TextureVectorView<float> data_;
    Bounds<float>            bounds_;

    TextureDimView() = default;
    TextureDimView(TextureVectorView<float> data, const Bounds<float>& bounds) :
        data_(data), bounds_(bounds)
    {}

    RTAC_HOSTDEVICE uint32_t      size()   const { return data_.size(); }
    RTAC_HOSTDEVICE Bounds<float> bounds() const { return bounds_;      }
    RTAC_HOSTDEVICE const TextureDimView& view() const { return *this; }

    #ifdef RTAC_CUDACC
    RTAC_HOSTDEVICE float index_to_value(uint32_t idx) const { return data_[idx]; }
    #endif //RTAC_CUDACC
};

class TextureDim : public DimExpression<TextureDim>
{
    protected:

    Bounds<float>        bounds_;
    HostVector<float>    data_;
    TextureVector<float> deviceData_;

    public:

    TextureDim(const HostVector<float>& data) :
        bounds_({data.front(), data.back()}), data_(data), deviceData_(data_)
    {}

    uint32_t      size()   const { return data_.size(); }
    const float*  data()   const { return data_.data(); }
    Bounds<float> bounds() const { return bounds_;      }
    float index_to_value(uint32_t idx) const { return data_[idx]; }


    TextureDimView view() const {
        return TextureDimView(deviceData_.view(), bounds_);
    }
};

} //namespace cuda
} //namespace rtac


#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_DIM_H_
