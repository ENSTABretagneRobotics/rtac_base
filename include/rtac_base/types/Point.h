#ifndef _DEF_RTAC_BASE_TYPES_POINT_H_
#define _DEF_RTAC_BASE_TYPES_POINT_H_

#include <iostream>

namespace rtac { namespace types {

template <typename T>
struct Point2
{
    T x;
    T y;
};

template <typename T>
struct Point3
{
    T x;
    T y;
    T z;
};

}; //namespace types
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Point2<T>& p)
{
    os << p.x << " " << p.y;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Point3<T>& p)
{
    os << p.x << " " << p.y << " " << p.z;
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POINT_H_
