#ifndef _DEF_RTAC_BASE_TYPES_IMAGE_H_
#define _DEF_RTAC_BASE_TYPES_IMAGE_H_

#include <iostream>
#include <vector>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/HostVector.h>

namespace rtac {

template <typename PixelT, template <typename> class ContainerT = HostVector>
class Image
{
    public:

    using value_type = PixelT;
    using Container  = ContainerT<PixelT>;
    using Shape      = rtac::Shape<uint32_t>; // using uint32_t for best cuda performances

    protected:

    Shape     shape_;
    Container data_;

    public:

    Image() : shape_({0,0}) {}
    Image(const Shape& shape) : shape_(shape), data_(shape.area()) {}
    Image(const Shape& shape, const Container& data) : shape_(shape), data_(data) {}
    template <typename T, template<typename> class C>
    Image(const Image<T,C>& other) : shape_(other.shape()), data_(other.container()) {}

    template <typename T, template<typename> class C> RTAC_HOSTDEVICE
    Image<PixelT,ContainerT>& operator=(const Image<T,C>& other) {
        shape_ = other.shape();
        data_  = other.container();
        return *this;
    }

    void resize(const Shape& shape) {
        data_.resize(shape.area());
        shape_ = shape;
    }

    RTAC_HOSTDEVICE const value_type* data()  const { return data_.data();  }
    RTAC_HOSTDEVICE value_type*       data()        { return data_.data();  }

    RTAC_HOSTDEVICE const Container& container() const { return data_; }
    RTAC_HOSTDEVICE       Container& container()       { return data_; }

    RTAC_HOSTDEVICE uint32_t     width()  const { return shape_.width;  }
    RTAC_HOSTDEVICE uint32_t     height() const { return shape_.height; }
    RTAC_HOSTDEVICE const Shape& shape()  const { return shape_; }
    RTAC_HOSTDEVICE auto         size()   const { return data_.size(); }

    RTAC_HOSTDEVICE PixelT  operator[](std::size_t idx) const { return data_[idx]; }
    RTAC_HOSTDEVICE PixelT& operator[](std::size_t idx)       { return data_[idx]; }
    RTAC_HOSTDEVICE PixelT  operator()(std::size_t h, std::size_t w) const {
        return data_[this->width()*h + w];
    }
    RTAC_HOSTDEVICE PixelT& operator()(std::size_t h, std::size_t w) {
        return data_[this->width()*h + w];
    }

    Image<const PixelT, VectorView> const_view() const { return this->view(); }
    Image<const PixelT, VectorView> view() const {
        return Image<const PixelT,VectorView>(this->shape(),
            VectorView<const PixelT>(data_.size(), data_.data()));
    }
    Image<PixelT, VectorView> view() {
        return Image<PixelT,VectorView>(this->shape(),
            VectorView<PixelT>(data_.size(), data_.data()));
    }
};

template <typename PixelT>
using ImageView = Image<PixelT, VectorView>;

}; //namespace rtac

template <typename T, template<typename> class C>
inline std::ostream& operator<<(std::ostream& os, const rtac::Image<T,C>& img)
{
    os << "Image (" << img.width() << 'x' << img.height() << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_IMAGE_H_

