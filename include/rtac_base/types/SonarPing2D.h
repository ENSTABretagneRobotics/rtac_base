#ifndef _DEF_RTAC_BASE_TYPES_SONAR_PING_2D_H_
#define _DEF_RTAC_BASE_TYPES_SONAR_PING_2D_H_

#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/Image.h>

namespace rtac {

/**
 * This is a container type to be used in all of rtac sonar related algorithms.
 *
 * This brings a form of (internal) standardization.
 */
template <typename T, template<typename>class VectorT>
class SonarPing2D
{
    public:

    using value_type = T;

    protected:

    Image<T,VectorT> pingData_;
    VectorT<float>   bearings_;
    Bounds<float,1>  bearingBounds_;
    Bounds<float,1>  rangeBounds_;

    public:

    SonarPing2D(const Shape<uint32_t>& shape = {0,0});
    SonarPing2D(const Image<T,VectorT>& pingData,
                const VectorT<float>&   bearings,
                const Bounds<float,1>&  bearingBounds,
                const Bounds<float,1>&  rangeBounds);

    template <template<typename>class VectorT2>
    SonarPing2D(const SonarPing2D<T,VectorT2>& other) : SonarPing2D() { *this = other; }
    template <template<typename>class VectorT2>
    SonarPing2D<T,VectorT>& operator=(const SonarPing2D<T,VectorT2>& other);

    void resize(const Shape<uint32_t>& shape);

    Shape<uint32_t> shape()  const { return pingData_.shape(); }
    uint32_t width()         const { return pingData_.width;  }
    uint32_t height()        const { return pingData_.height; }
    uint64_t size()          const { return this->width()*this->height(); }
    uint32_t bearing_count() const { return this->width();  }
    uint32_t height_count()  const { return this->height(); }

    const Image<T,VectorT>& ping_data() const { return pingData_; }
    const VectorT<float>&   bearings()  const { return bearings_; }

    void set_ping_data(const std::vector<T>& data);
    void set_bearings(const std::vector<float>& data);

    Bounds<float,1>   bearing_bounds() const { return bearingBounds_; }
    Bounds<float,1>   range_bounds()   const { return rangeBounds_;   }

    VectorView<T>    bearings()       { return VectorView<T>(bearings_); }
    ImageView<T>     ping_data()      { return pingData_.view();         }
    Bounds<float,1>& bearing_bounds() { return bearingBounds_;           }
    Bounds<float,1>& range_bounds()   { return rangeBounds_;             }

};

template <typename T, template<typename>class V>
SonarPing2D<T,V>::SonarPing2D(const Shape<uint32_t>& shape) :
    pingData_(shape),
    bearings_(shape.width),
    bearingBounds_(Bounds<float,1>::Zero()),
    rangeBounds_(Bounds<float,1>::Zero())
{}

template <typename T, template<typename>class V>
SonarPing2D<T,V>::SonarPing2D(const Image<T,V>& pingData,
                              const V<float>&   bearings,
                              const Bounds<float,1>&  bearingBounds,
                              const Bounds<float,1>&  rangeBounds) :
    pingData_(pingData),
    bearings_(bearings),
    bearingBounds_(bearingBounds),
    rangeBounds_(rangeBounds)
{}

template <typename T, template<typename>class V> template <template<typename>class V2>
SonarPing2D<T,V>& SonarPing2D<T,V>::operator=(const SonarPing2D<T,V2>& other)
{
    this->pingData_      = other.ping_data();
    this->bearings_      = other.bearings();
    this->bearingBounds_ = other.bearing_bounds();
    this->rangeBounds_   = other.range_bounds();
    return *this;
}

template <typename T, template<typename>class V>
void SonarPing2D<T,V>::resize(const Shape<uint32_t>& shape)
{
    pingData_.resize(shape);
    bearings_.resize(this->width());
}

template <typename T, template<typename>class V>
void SonarPing2D<T,V>::set_ping_data(const std::vector<T>& data)
{
    if(data.size() != this->size()) {
        std::ostringstream oss;
        oss << "rtac::SonarPing2D::set_ping_data : invalid size.";
        throw std::runtime_error(oss.str());
    }
    pingData_.container() = data;
}

template <typename T, template<typename>class V>
void SonarPing2D<T,V>::set_bearings(const std::vector<float>& data)
{
    if(data.size() != this->width()) {
        std::ostringstream oss;
        oss << "rtac::SonarPing2D::set_bearings : invalid size.";
        throw std::runtime_error(oss.str());
    }
    bearings_ = data;
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_SONAR_PING_H_

