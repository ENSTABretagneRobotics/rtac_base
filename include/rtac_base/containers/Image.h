#ifndef _DEF_RTAC_BASE_TYPES_IMAGE_H_
#define _DEF_RTAC_BASE_TYPES_IMAGE_H_

#include <iostream>
#include <vector>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/HostVector.h>

namespace rtac {

template <class Derived> class ImageExpression;
template <typename T>    class ImageView;

template <typename T>
ImageView<T> make_image_view(uint32_t width, uint32_t height, T* data, uint32_t step) {
    return ImageView<T>(width, height, data, step);
}
template <typename T>
ImageView<T> make_image_view(uint32_t width, uint32_t height, T* data) {
    return ImageView<T>(width, height, data, width);
}
template <typename T>
ImageView<const T> make_image_view(uint32_t width, uint32_t height,
                                   const T* data, uint32_t step) {
    return ImageView<const T>(width, height, data, step);
}
template <typename T>
ImageView<const T> make_image_view(uint32_t width, uint32_t height, const T* data) {
    return ImageView<const T>(width, height, data, width);
}

template <class Derived>
auto make_view(const ImageExpression<Derived>& img) {
    return make_image_view(img.width(), img.height(), img.data(), img.step());
}

template <class Derived>
auto make_view(ImageExpression<Derived>& img) {
    return make_image_view(img.width(), img.height(), img.data(), img.step());
}

template <class Derived>
struct ImageExpression
{
    RTAC_HOSTDEVICE const Derived* cast() const {
        return reinterpret_cast<const Derived*>(this);
    }
    RTAC_HOSTDEVICE Derived* cast() {
        return reinterpret_cast<Derived*>(this);
    }

    RTAC_HOSTDEVICE uint32_t size() const { return this->width()*this->height(); }

    RTAC_HOSTDEVICE auto  operator[](uint32_t idx) const { return this->data()[idx]; }
    RTAC_HOSTDEVICE auto& operator[](uint32_t idx)       { return this->data()[idx]; }
    RTAC_HOSTDEVICE auto  operator()(uint32_t h, uint32_t w) const {
        return this->data()[this->step()*h + w];
    }
    RTAC_HOSTDEVICE auto& operator()(uint32_t h, uint32_t w) {
        return this->data()[this->step()*h + w];
    }

    RTAC_HOSTDEVICE auto view() const {
        return make_image_view(this->width(), this->height(), this->data(), this->step());
    }
    RTAC_HOSTDEVICE auto view() {
        return make_image_view(this->width(), this->height(), this->data(), this->step());
    }

    // These 5 methods have to be reimplemented in subsclasses.
    RTAC_HOSTDEVICE const auto* data()   const { return this->cast()->data();   }
    RTAC_HOSTDEVICE auto*       data()         { return this->cast()->data();   }
    RTAC_HOSTDEVICE uint32_t    width()  const { return this->cast()->width();  }
    RTAC_HOSTDEVICE uint32_t    height() const { return this->cast()->height(); }
    RTAC_HOSTDEVICE uint32_t    step()   const { return this->cast()->step();   }
};

template <typename T>
class ImageView : public ImageExpression<ImageView<T>>
{
    public:

    using value_type = T;

    protected:

    T*       data_;
    uint32_t width_;
    uint32_t height_;
    uint32_t step_;

    public:

    RTAC_HOSTDEVICE ImageView(uint32_t width, uint32_t height,
                              T* data, uint32_t step) :
        data_(data), width_(width), height_(height), step_(step)
    {}
    RTAC_HOSTDEVICE ImageView(uint32_t width, uint32_t height, T* data) :
        ImageView<T>(width, height, data, width)
    {}
    template <class Derived>
    RTAC_HOSTDEVICE ImageView(ImageExpression<Derived>& other) { *this = other; }
    template <class Derived>
    RTAC_HOSTDEVICE ImageView<T> operator=(ImageExpression<Derived>& other) {
        data_   = other.data();   width_ = other.width();
        height_ = other.height(); step_  = other.step();
    }

    RTAC_HOSTDEVICE const T* data() const { return data_; }
    RTAC_HOSTDEVICE T* data() { return data_; }

    RTAC_HOSTDEVICE uint32_t width()  const { return width_;  }
    RTAC_HOSTDEVICE uint32_t height() const { return height_; }
    RTAC_HOSTDEVICE uint32_t step()   const { return step_;   }
};

template <typename T>
class ImageView<const T> : public ImageExpression<ImageView<const T>>
{
    public:

    using value_type = T;

    protected:

    const T* data_;
    uint32_t width_;
    uint32_t height_;
    uint32_t step_;

    public:

    RTAC_HOSTDEVICE ImageView(uint32_t width, uint32_t height,
                              const T* data, uint32_t step) :
        data_(data), width_(width), height_(height), step_(step)
    {}
    RTAC_HOSTDEVICE ImageView(uint32_t width, uint32_t height, const T* data) :
        ImageView<const T>(width, height, data, width)
    {}
    template <class Derived>
    RTAC_HOSTDEVICE ImageView(const ImageExpression<Derived>& other) { *this = other; }
    template <class Derived>
    RTAC_HOSTDEVICE ImageView<const T> operator=(const ImageExpression<Derived>& other) {
        data_   = other.data();   width_ = other.width();
        height_ = other.height(); step_  = other.step();
    }

    RTAC_HOSTDEVICE const T* data()   const { return data_; }
    RTAC_HOSTDEVICE uint32_t width()  const { return width_;  }
    RTAC_HOSTDEVICE uint32_t height() const { return height_; }
    RTAC_HOSTDEVICE uint32_t step()   const { return step_;   }
};

template <typename T, template <typename> class ContainerT = HostVector>
class Image : public ImageExpression<Image<T,ContainerT>>
{
    public:

    using value_type = T;
    using Container  = ContainerT<T>;
    using Shape      = rtac::Shape<uint32_t>; // using uint32_t for best cuda performances

    protected:

    Shape     shape_;
    Container data_;

    public:

    Image() : shape_({0,0}) {}
    Image(const Shape& shape) : shape_(shape), data_(shape.area()) {}
    Image(const Shape& shape, const Container& data) : shape_(shape), data_(data) {}
    template <template<typename> class C>
    Image(const Image<T,C>& other) : shape_(other.shape()), data_(other.container()) {}

    template <template<typename> class C> RTAC_HOSTDEVICE
    Image<T,ContainerT>& operator=(const Image<T,C>& other) {
        shape_ = other.shape();
        data_  = other.container();
        return *this;
    }

    void resize(const Shape& shape) {
        data_.resize(shape.area());
        shape_ = shape;
    }

    RTAC_HOSTDEVICE const Container& container() const { return data_; }
    RTAC_HOSTDEVICE       Container& container()       { return data_; }
    RTAC_HOSTDEVICE const Shape& shape()  const { return shape_;        }


    RTAC_HOSTDEVICE const T* data()   const { return data_.data();  }
    RTAC_HOSTDEVICE T*       data()         { return data_.data();  }
    RTAC_HOSTDEVICE uint32_t width()  const { return shape_.width;  }
    RTAC_HOSTDEVICE uint32_t height() const { return shape_.height; }
    RTAC_HOSTDEVICE uint32_t step()   const { return shape_.width;  }
};

}; //namespace rtac

template <typename T, template<typename> class C>
inline std::ostream& operator<<(std::ostream& os, const rtac::Image<T,C>& img)
{
    os << "Image (" << img.width() << 'x' << img.height() << ')';
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const rtac::ImageView<T>& img)
{
    os << "ImageView (" << img.width() << 'x' << img.height() << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_IMAGE_H_

