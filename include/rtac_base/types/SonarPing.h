#ifndef _DEF_RTAC_BASE_TYPES_SONAR_PING_H_
#define _DEF_RTAC_BASE_TYPES_SONAR_PING_H_

#include <rtac_base/types/Bounds.h>

namespace rtac {

template <class Derived>
struct SonarPingExpression2D
{
    const Derived* cast() const { return static_cast<const Derived*>(this); }
          Derived* cast()       { return static_cast<Derived*>(this);       }

    unsigned int bearing_count() const { return this->cast()->bearing_count(); }
    unsigned int bearing_min()   const { return this->cast()->bearing_min();   }
    unsigned int bearing_max()   const { return this->cast()->bearing_max();   }
    float bearing(unsigned int idx) const { return this->cast()->bearing(idx); }

    unsigned int range_count() const { return this->cast()->range_count(); }
    float        range_min()   const { return this->cast()->range_min();   }
    float        range_max()   const { return this->cast()->range_max();   }
    float range(unsigned int idx) const { return this->cast()->range(idx); }
};

} //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_SONAR_PING_H_
