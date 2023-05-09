#ifndef _DEF_RTAC_BASE_TYPES_SONAR_PING_H_
#define _DEF_RTAC_BASE_TYPES_SONAR_PING_H_

#include <iosfwd>

#include <rtac_base/types/TypeInfo.h>
#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Linspace.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/Image.h>

namespace rtac {

class PingBase2D
{
    protected:

    Linspace<float> ranges_;

    Bounds<float>   bearingBounds_;
    unsigned int    bearingCount_;

    ScalarId scalarType_;

    public:

    RTAC_HD_GENERIC PingBase2D(ScalarId scalarType) : 
        ranges_(0,1,0),
        bearingBounds_(-1,1),
        bearingCount_(0),
        scalarType_(scalarType)
    {}
    
    RTAC_HD_GENERIC PingBase2D(ScalarId scalarType,
                               const Linspace<float>& ranges,
                               const Bounds<float>& bearingBounds,
                               unsigned int bearingCount) :
        ranges_(ranges),
        bearingBounds_(bearingBounds),
        bearingCount_(bearingCount),
        scalarType_(scalarType)
    {}

    RTAC_HD_GENERIC PingBase2D(const PingBase2D&)            = default;
    RTAC_HD_GENERIC PingBase2D& operator=(const PingBase2D&) = default;

    RTAC_HD_GENERIC unsigned int bearing_count()          const { return bearingCount_;        }
    RTAC_HD_GENERIC float        bearing_min()            const { return bearingBounds_.lower; }
    RTAC_HD_GENERIC float        bearing_max()            const { return bearingBounds_.upper; }
    RTAC_HD_GENERIC const Bounds<float>& bearing_bounds() const { return bearingBounds_;       }
 
    RTAC_HD_GENERIC const Linspace<float>& ranges()       const { return ranges_;          }
    RTAC_HD_GENERIC unsigned int range_count()            const { return ranges_.size();   }
    RTAC_HD_GENERIC float        range_min()              const { return ranges_.lower();  }
    RTAC_HD_GENERIC float        range_max()              const { return ranges_.upper();  }
    RTAC_HD_GENERIC float range(unsigned int idx)         const { return ranges_[idx];     }
    RTAC_HD_GENERIC const Bounds<float>& range_bounds()   const { return ranges_.bounds(); }

    RTAC_HD_GENERIC unsigned int width()  const { return this->bearing_count();        }
    RTAC_HD_GENERIC unsigned int height() const { return this->range_count();          }
    RTAC_HD_GENERIC unsigned int size()   const { return this->width()*this->height(); }

    ScalarId scalar_type() const { return scalarType_; }
};

template <typename T,
          template<typename>class VectorT = HostVector>
class Ping2D : public PingBase2D//, public PingExpression2D<Ping2D<T, VectorT>>
{
    protected:

    VectorT<float> bearings_;
    VectorT<T>     pingData_;

    public:

    Ping2D() : PingBase2D(GetScalarId<T>::value) {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings) :
        PingBase2D(GetScalarId<T>::value,
                   ranges,
                   Bounds<float>(bearings.front(), bearings.back()),
                   bearings.size()),
        bearings_(bearings),
        pingData_(ranges.size()*bearings.size())
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings,
           const VectorT2<T>& pingData) :
        PingBase2D(GetScalarId<T>::value,
                   ranges,
                   Bounds<float>(bearings.front(), bearings.back()),
                   bearings.size()),
        bearings_(bearings),
        pingData_(pingData)
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings,
           const Bounds<float>& bearingBounds,
           const VectorT2<T>& pingData) :
        PingBase2D(GetScalarId<T>::value, ranges, bearingBounds, bearings.size()),
        bearings_(bearings),
        pingData_(pingData)
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Ping2D<T, VectorT2>& other) :
        PingBase2D(other),
        bearings_(other.bearings()),
        pingData_(other.ping_data_container())
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           VectorT2<float>&& bearings,
           VectorT2<T>&& pingData) :
        PingBase2D(GetScalarId<T>::value,
                   ranges,
                   Bounds<float>(bearings.front(), bearings.back()),
                   bearings.size()),
        bearings_(std::move(bearings)),
        pingData_(std::move(pingData))
    {}

    Ping2D(Ping2D<T, VectorT>&&)                      = default;
    Ping2D<T,VectorT>& operator=(Ping2D<T,VectorT>&&) = default;

    RTAC_HD_GENERIC const VectorT<float>& bearings() const { return bearings_;      }
    RTAC_HD_GENERIC float bearing(unsigned int idx)  const { return bearings_[idx]; }
 
    RTAC_HD_GENERIC const T* ping_data() const { return pingData_.data(); }
    RTAC_HD_GENERIC       T* ping_data()       { return pingData_.data(); }

    RTAC_HD_GENERIC const VectorT<T>& ping_data_container() const { return pingData_; }
    RTAC_HD_GENERIC       VectorT<T>& ping_data_container()       { return pingData_; }

    RTAC_HD_GENERIC const T& operator()(unsigned int r, unsigned int b) const { 
        return pingData_[this->bearing_count()*r + b];
    }
    RTAC_HD_GENERIC T& operator()(unsigned int r, unsigned int b) { 
        return pingData_[this->bearing_count()*r + b];
    }


    RTAC_HD_GENERIC Ping2D<T,ConstVectorView> view() const {
        return Ping2D<T,ConstVectorView>(ranges_,
                                         bearings_.view(),
                                         bearingBounds_,
                                         pingData_.view());
    }
    RTAC_HD_GENERIC Ping2D<T,VectorView> view() {
        return Ping2D<T,VectorView>(ranges_,
                                    bearings_.view(),
                                    bearingBounds_,
                                    pingData_.view());
    }

    RTAC_HD_GENERIC Ping2D<T,VectorT>& set_ranges(const Linspace<float>& ranges,
                                                  bool resizeData = true)
    {
        this->ranges_ = ranges;
        if(resizeData) {
            pingData_.resize(this->size());
        }
        return *this;
    }

    RTAC_HD_WARNING template <template<typename>class VectorT2>
    RTAC_HOSTDEVICE Ping2D<T,VectorT>& set_bearings(const VectorT2<float>& bearings,
                                                    bool resizeData = true)
    {
        this->bearings_      = bearings;
        this->bearingBounds_ = Bounds<float>(bearings.front(), bearings.back());
        this->bearingCount_  = bearings_.size();
        if(resizeData) {
            pingData_.resize(this->size());
        }
        return *this;
    }

    RTAC_HD_GENERIC auto image_view() { return make_image_view(this->width(),
                                                               this->height(), 
                                                               this->ping_data()); }
    RTAC_HD_GENERIC auto image_view() const { return make_image_view(this->width(),
                                                                     this->height(), 
                                                                     this->ping_data()); }
};

template <typename T> using PingView2D      = Ping2D<T,VectorView>;
template <typename T> using PingConstView2D = Ping2D<T,ConstVectorView>;

} //namespace rtac

//template <class Derived>
//std::ostream& operator<<(std::ostream& os, const rtac::PingExpression2D<Derived>& ping)
inline std::ostream& operator<<(std::ostream& os, const rtac::PingBase2D& ping)
{
    os << "ranges (" <<  ping.range_count() << ") : "
       << ping.range_min() << " -> " << ping.range_max()
       << ", bearings (" <<  ping.bearing_count() << ") : "
       << ping.bearing_min() << " -> " << ping.bearing_max();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SONAR_PING_H_
