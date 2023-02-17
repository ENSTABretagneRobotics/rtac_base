#ifndef _DEF_RTAC_BASE_CUDA_TEXTURE_VECTOR_H_
#define _DEF_RTAC_BASE_CUDA_TEXTURE_VECTOR_H_

#include <vector>

#include <rtac_base/containers/HostVector.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>

namespace rtac { namespace cuda {

template <typename T>
struct TextureVectorView
{
    cudaTextureObject_t data_;
    const std::size_t   size_;

    TextureVectorView() = default;
    TextureVectorView<T>& operator=(const TextureVectorView<T>&) = default;
    
    RTAC_HOSTDEVICE std::size_t size() const { return size_; }
    #ifdef RTAC_CUDACC
    __host__ __device__ T operator[](int idx) const { return tex1Dfetch<T>(data_, idx); }
    #endif
};

template <typename T>
class TextureVector
{
    public:

    using value_type = T;

    protected:

    DeviceVector<T>     data_;
    cudaTextureObject_t textureHandle_;

    public:

    TextureVector() {}
    TextureVector(std::size_t size)                { this->resize(size); }
    TextureVector(std::size_t size, const T* data) { this->set_data(size, data); }
    TextureVector(const TextureVector<T>& other)   { *this = other; }
    TextureVector(const DeviceVector<T>& other)    { *this = other; }
    TextureVector(const HostVector<T>& other)      { *this = other; }
    TextureVector(const std::vector<T>& other)     { *this = other; }
    ~TextureVector() { this->clear(); }

    TextureVector<T>& operator=(const TextureVector<T>& other);
    TextureVector<T>& operator=(const DeviceVector<T>& other);
    TextureVector<T>& operator=(const HostVector<T>& other);
    TextureVector<T>& operator=(const std::vector<T>& other);

    void resize(std::size_t size);
    void update_texture_handle();
    void destroy_texture_handle();
    void clear();

    std::size_t size() const { return data_.size(); }
    const T*    data() const { return data_.data(); }
          T*    data()       { return data_.data(); }
    const DeviceVector<T>& container() const { return data_; }
    TextureVectorView<T> view() const;

    void copy_from_host(std::size_t size, const T* data);
    void copy_from_device(std::size_t size, const T* data);
};

template <typename T>
TextureVector<T>& TextureVector<T>::operator=(const TextureVector<T>& other)
{
    data_ = other.data_;
    this->update_texture_handle();
    return *this;
}

template <typename T>
TextureVector<T>& TextureVector<T>::operator=(const DeviceVector<T>& other)
{
    data_ = other;
    this->update_texture_handle();
    return *this;
}

template <typename T>
TextureVector<T>& TextureVector<T>::operator=(const HostVector<T>& other)
{
    data_ = other;
    this->update_texture_handle();
    return *this;
}

template <typename T>
TextureVector<T>& TextureVector<T>::operator=(const std::vector<T>& other)
{
    data_ = other;
    this->update_texture_handle();
    return *this;
}

template <typename T>
void TextureVector<T>::resize(std::size_t size)
{
    if(size > data_.capacity()) {
        data_.resize();
        this->update_texture_handle();
    }
}

template <typename T>
void TextureVector<T>::update_texture_handle()
{
    this->destroy_texture_handle();

    // ignored if no data.
    if(this->size() == 0)
        return;

    cudaResourceDesc resourceDescription = {}; // zero initialization
    resourceDescription.resType                = cudaResourceTypeLinear;
    resourceDescription.res.linear.devPtr      = (void*)data_.data();
    resourceDescription.res.linear.desc        = cudaCreateChannelDesc<T>();
    resourceDescription.res.linear.sizeInBytes = sizeof(T)*data_.size();

    cudaTextureDesc textureDescription  = {};
    textureDescription.addressMode[0]   = cudaAddressModeClamp; // ignored in linear mode ?
    textureDescription.addressMode[1]   = cudaAddressModeClamp; // ignored in linear mode ?
    textureDescription.addressMode[2]   = cudaAddressModeClamp; // ignored in linear mode ?
    textureDescription.filterMode       = cudaFilterModePoint;  // ignored in linear mode ?
    textureDescription.readMode         = cudaReadModeElementType;
    textureDescription.normalizedCoords = 0;                    // read with integers


    CUDA_CHECK( cudaCreateTextureObject(&textureHandle_, &resourceDescription,
                                        &textureDescription, nullptr) );
}

template <typename T>
void TextureVector<T>::destroy_texture_handle()
{
    CUDA_CHECK( cudaDestroyTextureObject(textureHandle_) );
    textureHandle_ = 0;
}

template <typename T>
void TextureVector<T>::clear()
{
    this->destroy_texture_handle();
    data_.clear();
}

template <typename T>
void TextureVector<T>::copy_from_host(std::size_t size, const T* data)
{
    this->resize(size);
    data_->copy_from_host(size, data);
}

template <typename T>
void TextureVector<T>::copy_from_device(std::size_t size, const T* data)
{
    this->resize(size);
    data_->copy_from_device(size, data);
}

template <typename T>
TextureVectorView<T> TextureVector<T>::view() const 
{ 
    return TextureVectorView<T>{textureHandle_, this->size()};
}

} //namespace cuda
} //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_TEXTURE_VECTOR_H_
