#ifndef _DEF_RTAC_CUDA_TEXTURE_2D_H_
#define _DEF_RTAC_CUDA_TEXTURE_2D_H_

#include <iostream>
#include <vector>
#include <limits>

#include <cuda_runtime.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/DeviceVector.h>

namespace rtac { namespace cuda {

template <typename T>
class Texture2D
{
    public:

    using value_type = T;

    static cudaChannelFormatDesc channel_description();
    static cudaTextureDesc       default_texture_description();

    // These are aliases to CUDA texture enumerations
    using FilterMode = cudaTextureFilterMode;
    static const cudaTextureFilterMode FilterNearest = cudaFilterModePoint;
    static const cudaTextureFilterMode FilterLinear  = cudaFilterModeLinear;

    using WrapMode = cudaTextureAddressMode;
    static const cudaTextureAddressMode WrapRepeat = cudaAddressModeWrap;
    static const cudaTextureAddressMode WrapClamp  = cudaAddressModeClamp;
    static const cudaTextureAddressMode WrapMirror = cudaAddressModeMirror;
    static const cudaTextureAddressMode WrapBorder = cudaAddressModeBorder;

    using ReadMode = cudaTextureReadMode;
    static const cudaTextureReadMode ReadElementType     = cudaReadModeElementType;
    static const cudaTextureReadMode ReadNormalizedFloat = cudaReadModeNormalizedFloat;

    // Helpers to generate helful textures such as a checkerboard.
    static Texture2D<T> checkerboard(size_t width, size_t height,
                                     const T& black, const T& white,
                                     unsigned int oversampling = 1);

    protected:
    
    size_t              width_;
    size_t              height_;
    cudaArray*          data_;
    cudaTextureDesc     description_;
    cudaTextureObject_t textureHandle_;

    void free_data();
    void destroy_texture_handle();
    void update_texture_handle();

    public:

    Texture2D();
    ~Texture2D();

    void allocate_data(size_t width, size_t height);
    void set_image(size_t width, size_t height, const T* data);
    void set_subimage(size_t width,   size_t height, 
                      size_t wOffset, size_t hOffset, 
                      const T* data);

    size_t width()  const;
    size_t height() const;

    cudaTextureDesc description() const;

    cudaTextureObject_t texture();
    operator cudaTextureObject_t();

    cudaTextureObject_t texture()  const;
    operator cudaTextureObject_t() const;


    // Setters for texture fetch configuration (the way the texture will be
    // read when used on the device.
    void set_filter_mode(FilterMode mode, bool updateTexture = true);
    void set_wrap_mode(WrapMode xyWrap, bool updateTexture = true);
    void set_wrap_mode(WrapMode xWrap, WrapMode yWrap, bool updateTexture = true);
    void set_wrap_mode(WrapMode xWrap, WrapMode yWrap, WrapMode zWrap,
                       bool updateTexture = true);
    void set_read_mode(ReadMode mode, bool updateTexture = true);
    void use_trilinear_optimization(bool use, bool updateTexture = true);
    void use_normalized_coordinates(bool use, bool updateTexture = true);
    void set_max_anisotropy(unsigned int maxLevel, bool updateTexture = true);
    void perform_srgb_linear_conversion(bool doConversion, bool updateTexture = true);
    void set_mipmap_parameters(FilterMode mipmapFilterMode,
                               float mipmapLevelBias,
                               float minMipmapLevelClamp,
                               float maxMipmapLevelClamp,
                               bool updateTexture = true);
    void disable_mipmap(bool updateTexture = true);
};

//Implementation
template <typename T>
Texture2D<T>::Texture2D() :
    width_(0),
    height_(0),
    data_(nullptr),
    description_(default_texture_description()),
    textureHandle_(0)
{
}

template <typename T>
cudaChannelFormatDesc Texture2D<T>::channel_description()
{
    return cudaCreateChannelDesc<T>();
}

template <typename T>
Texture2D<T> Texture2D<T>::checkerboard(size_t width, size_t height,
                                        const T& black, const T& white,
                                        unsigned int oversampling)
{
    Texture2D<T> res;

    width  *= oversampling;
    height *= oversampling;

    // Generating the checkerboard.
    std::vector<T> data(width*height);
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
            if((h / oversampling + w / oversampling) & 0x1)
                data[width*h + w] = white;
            else
                data[width*h + w] = black;
        }
    }
    
    // Configuring texture.
    res.set_filter_mode(FilterNearest, false);
    res.set_wrap_mode(WrapRepeat, false);
    
    // Uploading data to device.
    res.set_image(width, height, data.data());

    return res;
}

template <typename T>
cudaTextureDesc Texture2D<T>::default_texture_description()
{
    auto res = zero<cudaTextureDesc>();

    res.addressMode[0]   = WrapRepeat;
    res.addressMode[1]   = WrapRepeat;
    res.addressMode[2]   = WrapClamp;
    res.filterMode       = FilterNearest;
    res.readMode         = ReadElementType;
    res.normalizedCoords = 1;

    return res;
}

template <typename T>
Texture2D<T>::~Texture2D()
{
    this->free_data();
    this->destroy_texture_handle();
}

template <typename T>
void Texture2D<T>::free_data()
{
    if(data_) {
        CUDA_CHECK( cudaFreeArray(data_) );
        data_ = nullptr;
    }
}

template <typename T>
void Texture2D<T>::destroy_texture_handle()
{
    if(textureHandle_) {
        CUDA_CHECK( cudaDestroyTextureObject(textureHandle_) );
        textureHandle_ = 0;
    }
}

template <typename T>
void Texture2D<T>::update_texture_handle()
{
    // ignoring if no data
    if(width_ == 0 || height_ == 0 || data_ == nullptr)
        return;
    
    this->destroy_texture_handle();

    // Creating texture object pointing to allocated data.
    auto resourceDescription = zero<cudaResourceDesc>();
    resourceDescription.resType = cudaResourceTypeArray;
    // resourceDescription.array = data_; // in cuda docs but not working ?
    resourceDescription.res.array.array = data_;

    CUDA_CHECK( cudaCreateTextureObject(&textureHandle_, &resourceDescription,
                                        &description_, NULL) );
}

template <typename T>
void Texture2D<T>::allocate_data(size_t width, size_t height)
{
    // Not reallocating if dimensions did not changed.
    if(width == width_ && height == height)
        return;

    this->free_data();
    width_ = 0; height_ = 0;

    auto channelDesc = Texture2D<T>::channel_description();
    CUDA_CHECK( cudaMallocArray(&data_, &channelDesc, width, height) );

    width_  = width;
    height_ = height;
}

template <typename T>
void Texture2D<T>::set_image(size_t width, size_t height, const T* data)
{
    this->allocate_data(width, height);

    //CUDA_CHECK( cudaMemcpyToArray(data_, 0, 0, data,
    //                              sizeof(T)*width_*height_,
    //                              cudaMemcpyHostToDevice) );

    CUDA_CHECK( cudaMemcpy2DToArray(data_, 0, 0,
                                    data,
                                    sizeof(T)*width_,
                                    sizeof(T)*width_,
                                    height_,
                                    cudaMemcpyHostToDevice) );

    // A new texture handle creation seems to be necessary when data is changed.
    this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_subimage(size_t width,   size_t height, 
                                size_t wOffset, size_t hOffset, 
                                const T* data)
{
    if(width + wOffset > width_ || height + hOffset > height_) {
        std::ostringstream oss;
        oss << "Texture2D::set_subimage : sub image not contained within "
            << "previously existing allocatged storage. (Current W/H : "
            << width_ << "/" << height_ << ", size needed for subimage : "
            << width + wOffset << "/" << height + hOffset << ")";
        throw std::runtime_error(oss.str());
    }

    CUDA_CHECK( cudaMemcpy2DToArray(data_,
                                    sizeof(T)*wOffset, hOffset,
                                    data,
                                    sizeof(T)*width,
                                    sizeof(T)*width,
                                    height,
                                    cudaMemcpyHostToDevice) );

    // A new texture handle creation seems to be necessary when data is changed.
    this->update_texture_handle();
}

template <typename T>
size_t Texture2D<T>::width() const
{
    return width_;
}

template <typename T>
size_t Texture2D<T>::height() const
{
    return height_;
}

template <typename T>
struct cudaTextureDesc Texture2D<T>::description() const
{
    return description_;
}

template <typename T>
cudaTextureObject_t Texture2D<T>::texture()
{
    return textureHandle_;
}

template <typename T>
Texture2D<T>::operator cudaTextureObject_t()
{
    return textureHandle_;
}

template <typename T>
cudaTextureObject_t Texture2D<T>::texture() const
{
    return textureHandle_;
}

template <typename T>
Texture2D<T>::operator cudaTextureObject_t() const
{
    return textureHandle_;
}

// Setters for texture fetch configuration (the way the texture will be
// read when used on the device.
template <typename T>
void Texture2D<T>::set_filter_mode(FilterMode mode, bool updateTexture)
{
    description_.filterMode = mode;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_wrap_mode(WrapMode xyWrap, bool updateTexture)
{
    this->set_wrap_mode(xyWrap, xyWrap, WrapClamp, updateTexture);

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_wrap_mode(WrapMode xWrap, WrapMode yWrap, bool updateTexture)
{
    this->set_wrap_mode(xWrap, yWrap, WrapClamp, updateTexture);

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_wrap_mode(WrapMode xWrap, WrapMode yWrap, WrapMode zWrap,
                   bool updateTexture)
{
    description_.addressMode[0] = xWrap;
    description_.addressMode[1] = yWrap;
    description_.addressMode[2] = zWrap;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_read_mode(ReadMode mode, bool updateTexture)
{
    description_.readMode = mode;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::use_trilinear_optimization(bool use, bool updateTexture)
{
    if(use)
        description_.disableTrilinearOptimization = 0;
    else
        description_.disableTrilinearOptimization = 1;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::use_normalized_coordinates(bool use, bool updateTexture)
{
    if(use)
        description_.normalizedCoords = 1;
    else
        description_.normalizedCoords = 0;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_max_anisotropy(unsigned int maxLevel, bool updateTexture)
{
    description_.maxAnisotropy = maxLevel;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::perform_srgb_linear_conversion(bool doConversion, bool updateTexture)
{
    if(doConversion)
        description_.sRGB = 1;
    else
        description_.sRGB = 0;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::set_mipmap_parameters(FilterMode mipmapFilterMode,
                                         float mipmapLevelBias,
                                         float minMipmapLevelClamp,
                                         float maxMipmapLevelClamp,
                                         bool updateTexture)
{
    description_.mipmapFilterMode    = mipmapFilterMode;
    description_.mipmapLevelBias     = mipmapLevelBias;
    description_.minMipmapLevelClamp = minMipmapLevelClamp;
    description_.maxMipmapLevelClamp = maxMipmapLevelClamp;

    if(updateTexture)
        this->update_texture_handle();
}

template <typename T>
void Texture2D<T>::disable_mipmap(bool updateTexture)
{
    memset(&description_.mipmapFilterMode, 0, sizeof(description_.mipmapFilterMode));
    description_.mipmapLevelBias     = 0.0f;
    description_.minMipmapLevelClamp = 0.0f;
    description_.maxMipmapLevelClamp = 0.0f;

    if(updateTexture)
        this->update_texture_handle();
}


}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_CUDA_TEXTURE_2D_H_
