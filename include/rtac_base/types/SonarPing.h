#ifndef _DEF_RTAC_BASE_TYPES_SONAR_PING_H_
#define _DEF_RTAC_BASE_TYPES_SONAR_PING_H_

#include <iostream>

#include <rtac_base/types/Bounds.h>
#include <rtac_base/types/Linspace.h>
#include <rtac_base/containers/HostVector.h>

namespace rtac {

template <class Derived>
struct PingExpression2D
{
    const Derived* cast() const { return static_cast<const Derived*>(this); }
          Derived* cast()       { return static_cast<Derived*>(this);       }

    unsigned int bearing_count()    const { return this->cast()->bearing_count(); }
    float        bearing_min()      const { return this->cast()->bearing_min();   }
    float        bearing_max()      const { return this->cast()->bearing_max();   }
    float bearing(unsigned int idx) const { return this->cast()->bearing(idx);    }

    unsigned int range_count()    const { return this->cast()->range_count(); }
    float        range_min()      const { return this->cast()->range_min();   }
    float        range_max()      const { return this->cast()->range_max();   }
    float range(unsigned int idx) const { return this->cast()->range(idx);    }

    const auto* ping_data() const { return this->cast()->ping_data(); }
          auto* ping_data()       { return this->cast()->ping_data(); }

    const auto& operator()(unsigned int r, unsigned int b) const { 
        return this->ping_data()[this->bearing_count()*r + b];
    }
    auto& operator()(unsigned int r, unsigned int b) { 
        return this->ping_data()[this->bearing_count()*r + b];
    }
    unsigned int width()  const { return this->bearing_count(); }
    unsigned int height() const { return this->range_count();   }
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
    Ping2D(const Ping2D<T, VectorT2>& other) :
        ranges_(other.ranges()),
        bearingBounds_(other.bearing_min(), other.bearing_max()),
        bearingCount_(other.bearing_count()),
        bearings_(other.bearings()),
        pingData_(other.ping_container())
    {}

    const VectorT<float>& bearings() const { return bearings_;            }
    const Bounds<float>& bearing_bounds() const { return bearingBounds_; }
    unsigned int bearing_count()     const { return bearings_.size();     }
    float        bearing_min()       const { return bearingBounds_.lower; }
    float        bearing_max()       const { return bearingBounds_.upper; }
    float bearing(unsigned int idx)  const { return bearings_[idx];       }

    const Linspace<float>& ranges() const { return ranges_;         }
    unsigned int range_count()      const { return ranges_.size();  }
    float        range_min()        const { return ranges_.lower(); }
    float        range_max()        const { return ranges_.upper(); }
    float range(unsigned int idx)   const { return ranges_[idx];    }

    const T* ping_data() const { return pingData_.data(); }
          T* ping_data()       { return pingData_.data(); }

    const VectorT<T>& ping_container() const { return pingData_; }
          VectorT<T>& ping_container()       { return pingData_; }
};

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
