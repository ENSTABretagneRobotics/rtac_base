#ifndef _DEF_RTAC_BASE_TYPES_IMAGE_H_
#define _DEF_RTAC_BASE_TYPES_IMAGE_H_

#include <iostream>
#include <vector>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/type_utils.h>
#include <rtac_base/types/Shape.h>

namespace rtac { namespace types {

template <typename PixelT, template <typename> class ContainerT = std::vector>
class Image
{
    public:

    using value_type = PixelT;
    using Container  = ContainerT<PixelT>;
    using Shape      = rtac::types::Shape<std::size_t>;

    protected:

    Shape     shape_;
    Container data_;

    public:

    Image() : shape_({0,0}) {}
    Image(const Shape& shape) : shape_(shape), data_(shape.area()) {}
    template <typename T, template<typename> class C>
    Image(const Image<T,C>& other) : shape_(other.shape()), data_(other.data()) {}

    template <typename T, template<typename> class C>
    Image<PixelT,ContainerT>& operator=(const Image<T,C>& other) {
        shape_ = other.shape();
        data_  = other.data();
        return *this;
    }

    void resize(const Shape& shape) {
        data_.resize(shape.area());
        shape_ = shape;
    }

    const Container& data()  const { return data_;  }
    Container&       data()        { return data_;  }

    RTAC_HOSTDEVICE std::size_t  width()  const { return shape_.width;  }
    RTAC_HOSTDEVICE std::size_t  height() const { return shape_.height; }
    RTAC_HOSTDEVICE const Shape& shape()  const { return shape_; }
    RTAC_HOSTDEVICE auto         size()   const { return data_.size(); }

    RTAC_HOSTDEVICE PixelT  operator[](std::size_t idx) const;
    RTAC_HOSTDEVICE PixelT& operator[](std::size_t idx);
    RTAC_HOSTDEVICE PixelT  operator()(std::size_t h, std::size_t w) const;
    RTAC_HOSTDEVICE PixelT& operator()(std::size_t h, std::size_t w);
};

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

}; //namespace types
}; //namespace rtac

template <typename T, template<typename> class C>
inline std::ostream& operator<<(std::ostream& os, const rtac::types::Image<T,C>& img)
{
    os << "Image (" << img.width() << 'x' << img.height() << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_IMAGE_H_

