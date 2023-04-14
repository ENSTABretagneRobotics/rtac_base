#ifndef _DEF_RTAC_BASE_TYPES_SONAR_PING_H_
#define _DEF_RTAC_BASE_TYPES_SONAR_PING_H_

#include <iostream>

#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Linspace.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/VectorView.h>

namespace rtac {

template <class Derived>
struct PingExpression2D
{
    RTAC_HD_GENERIC const Derived* cast() const { return static_cast<const Derived*>(this); }
    RTAC_HD_GENERIC       Derived* cast()       { return static_cast<Derived*>(this);       }

    RTAC_HD_GENERIC unsigned int bearing_count()    const { return this->cast()->bearing_count(); }
    RTAC_HD_GENERIC float        bearing_min()      const { return this->cast()->bearing_min();   }
    RTAC_HD_GENERIC float        bearing_max()      const { return this->cast()->bearing_max();   }
    RTAC_HD_GENERIC float bearing(unsigned int idx) const { return this->cast()->bearing(idx);    }

    RTAC_HD_GENERIC unsigned int range_count()    const { return this->cast()->range_count(); }
    RTAC_HD_GENERIC float        range_min()      const { return this->cast()->range_min();   }
    RTAC_HD_GENERIC float        range_max()      const { return this->cast()->range_max();   }
    RTAC_HD_GENERIC float range(unsigned int idx) const { return this->cast()->range(idx);    }

    RTAC_HD_GENERIC const auto* ping_data() const { return this->cast()->ping_data(); }
    RTAC_HD_GENERIC       auto* ping_data()       { return this->cast()->ping_data(); }

    RTAC_HD_GENERIC const auto& operator()(unsigned int r, unsigned int b) const { 
        return this->ping_data()[this->bearing_count()*r + b];
    }
    RTAC_HD_GENERIC auto& operator()(unsigned int r, unsigned int b) { 
        return this->ping_data()[this->bearing_count()*r + b];
    }
    RTAC_HD_GENERIC unsigned int width()  const { return this->bearing_count(); }
    RTAC_HD_GENERIC unsigned int height() const { return this->range_count();   }
};


template <typename T,
          template<typename>class VectorT = HostVector>
class Ping2D : public PingExpression2D<Ping2D<T, VectorT>>
{
    protected:

    Linspace<float> ranges_;

    Bounds<float>   bearingBounds_;
    unsigned int    bearingCount_;
    VectorT<float>  bearings_;

    VectorT<T> pingData_;

    public:

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings) :
        ranges_(ranges),
        bearingBounds_(bearings.front(), bearings.back()),
        bearingCount_(bearings.size()),
        bearings_(bearings),
        pingData_(ranges.size()*bearings.size())
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings,
           const VectorT2<float>& pingData) :
        ranges_(ranges),
        bearingBounds_(bearings.front(), bearings.back()),
        bearingCount_(bearings.size()),
        bearings_(bearings),
        pingData_(pingData)
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Linspace<float>& ranges,
           const VectorT2<float>& bearings,
           const Bounds<float>& bearingBounds,
           const VectorT2<float>& pingData) :
        ranges_(ranges),
        bearingBounds_(bearingBounds),
        bearingCount_(bearings.size()),
        bearings_(bearings),
        pingData_(pingData)
    {}

    template <template<typename>class VectorT2>
    Ping2D(const Ping2D<T, VectorT2>& other) :
        ranges_(other.ranges()),
        bearingBounds_(other.bearing_min(), other.bearing_max()),
        bearingCount_(other.bearing_count()),
        bearings_(other.bearings()),
        pingData_(other.ping_data_container())
    {}

    RTAC_HD_GENERIC const VectorT<float>& bearings() const { return bearings_;            }
    RTAC_HD_GENERIC unsigned int bearing_count()     const { return bearings_.size();     }
    RTAC_HD_GENERIC float        bearing_min()       const { return bearingBounds_.lower; }
    RTAC_HD_GENERIC float        bearing_max()       const { return bearingBounds_.upper; }
    RTAC_HD_GENERIC float bearing(unsigned int idx)  const { return bearings_[idx];       }
    RTAC_HD_GENERIC const Bounds<float>& bearing_bounds() const { return bearingBounds_; }
 
    RTAC_HD_GENERIC const Linspace<float>& ranges() const { return ranges_;         }
    RTAC_HD_GENERIC unsigned int range_count()      const { return ranges_.size();  }
    RTAC_HD_GENERIC float        range_min()        const { return ranges_.lower(); }
    RTAC_HD_GENERIC float        range_max()        const { return ranges_.upper(); }
    RTAC_HD_GENERIC float range(unsigned int idx)   const { return ranges_[idx];    }
    RTAC_HD_GENERIC const Bounds<float>& range_bounds() const { return ranges_.bounds(); }

    RTAC_HD_GENERIC const T* ping_data() const { return pingData_.data(); }
    RTAC_HD_GENERIC       T* ping_data()       { return pingData_.data(); }

    RTAC_HD_GENERIC const VectorT<T>& ping_data_container() const { return pingData_; }
    RTAC_HD_GENERIC       VectorT<T>& ping_data_container()       { return pingData_; }

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
};

template <typename T> using PingView2D      = Ping2D<T,VectorView>;
template <typename T> using PingConstView2D = Ping2D<T,ConstVectorView>;

} //namespace rtac

template <class Derived> inline
std::ostream& operator<<(std::ostream& os, const rtac::PingExpression2D<Derived>& ping)
{
    os << "ranges (" <<  ping.range_count() << ") : "
       << ping.range_min() << " -> " << ping.range_max()
       << ", bearings (" <<  ping.bearing_count() << ") : "
       << ping.bearing_min() << " -> " << ping.bearing_max();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SONAR_PING_H_
