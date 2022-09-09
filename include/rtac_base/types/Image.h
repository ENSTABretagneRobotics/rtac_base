#ifndef _DEF_RTAC_BASE_TYPES_IMAGE_H_
#define _DEF_RTAC_BASE_TYPES_IMAGE_H_

#include <iostream>
#include <vector>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/type_utils.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/types/VectorView.h>

namespace rtac { namespace types {

template <typename PixelT, template <typename> class ContainerT>
class Image
{
    public:

    using value_type = PixelT;
    using Container  = ContainerT<PixelT>;
    using Shape      = rtac::types::Shape<uint32_t>;

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

    RTAC_HOSTDEVICE PixelT  operator[](std::size_t idx) const;
    RTAC_HOSTDEVICE PixelT& operator[](std::size_t idx);
    RTAC_HOSTDEVICE PixelT  operator()(std::size_t h, std::size_t w) const;
    RTAC_HOSTDEVICE PixelT& operator()(std::size_t h, std::size_t w);

    Image<const PixelT, VectorView> const_view() const { return this->view(); }
    Image<const PixelT, VectorView> view() const;
    Image<PixelT, VectorView>       view();
};

template <typename PixelT>
using ImageView = Image<PixelT, VectorView>;

template <typename T, template<typename> class C>
RTAC_HOSTDEVICE T Image<T,C>::operator[](std::size_t idx) const
{
    static_assert(is_subscriptable<Container>::value,
                  "rtac::types::Image : container is not subscriptable.");
    return data_[idx];
}

template <typename T, template<typename> class C>
RTAC_HOSTDEVICE T& Image<T,C>::operator[](std::size_t idx)
{
    static_assert(is_subscriptable<Container>::value,
                  "rtac::types::Image : container is not subscriptable.");
    return data_[idx];
}

template <typename T, template<typename> class C>
RTAC_HOSTDEVICE T Image<T,C>::operator()(std::size_t h, std::size_t w) const
{
    static_assert(is_subscriptable<Container>::value,
                  "rtac::types::Image : container is not subscriptable.");
    return data_[shape_.width*h + w];
}

template <typename T, template<typename> class C>
RTAC_HOSTDEVICE T& Image<T,C>::operator()(std::size_t h, std::size_t w)
{
    static_assert(is_subscriptable<Container>::value,
                  "rtac::types::Image : container is not subscriptable.");
    return data_[shape_.width*h + w];
}

template <typename T, template<typename> class C>
Image<const T, VectorView> Image<T,C>::view() const
{
    return Image<const T,VectorView>(this->shape(),
        VectorView<const T>(data_.size(), data_.data()));
}

template <typename T, template<typename> class C>
Image<T, VectorView> Image<T,C>::view()
{
    return Image<T,VectorView>(this->shape(), VectorView<T>(data_.size(), data_.data()));
}

}; //namespace types
}; //namespace rtac

template <typename T, template<typename> class C>
inline std::ostream& operator<<(std::ostream& os, const rtac::types::Image<T,C>& img)
{
    os << "Image (" << img.width() << 'x' << img.height() << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_IMAGE_H_

