#ifndef _DEF_RTAC_BASE_TYPES_SHAPE_H_
#define _DEF_RTAC_BASE_TYPES_SHAPE_H_

#include <iostream>

namespace rtac { namespace types {

template <typename T>
struct Shape
{
    public:

    T width;
    T height;

    template <typename RatioType = T>
    RatioType ratio()
    {
        return ((RatioType)width / height);
    }
};

}; //namespace types
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Shape<T>& shape)
{
    os << "width : " << shape.width << ", height : " << shape.height;
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SHAPE_H_
