#ifndef _DEF_RTAC_BASE_TYPES_POINT_H_
#define _DEF_RTAC_BASE_TYPES_POINT_H_

#include <iostream>

namespace rtac {

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

template <typename T>
struct Point4
{
    T x;
    T y;
    T z;
    T w;
};

}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Point2<T>& p)
{
    os << p.x << " " << p.y;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Point3<T>& p)
{
    os << p.x << " " << p.y << " " << p.z;
    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Point4<T>& p)
{
    os << p.x << " " << p.y << " " << p.z << " " << p.w;
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POINT_H_
