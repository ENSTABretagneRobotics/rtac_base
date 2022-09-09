#ifndef _DEF_RTAC_BASE_TYPES_SHAPE_H_
#define _DEF_RTAC_BASE_TYPES_SHAPE_H_

#include <iostream>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

/**
 * Represent the dimension of an image or another 2 dimensional buffer.
 *
 * @tparam Underlying scalar type.
 */
template <typename T>
struct Shape
{
    public:

    T width;
    T height;

    template <typename RatioType = T>
    RTAC_HOSTDEVICE RatioType ratio() const
    {
        return ((RatioType)width / height);
    }
    
    T area() const
    {
        return width * height;
    }

    template <typename SizeType = T>
    RTAC_HOSTDEVICE SizeType size() const
    {
        return width * height;
    }
    
    template <typename OtherT>
    RTAC_HOSTDEVICE operator Shape<OtherT>() const
    {
        return Shape<OtherT>{width, height};
    }
};

}; //namespace types
}; //namespace rtac

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const rtac::types::Shape<T>& shape)
{
    os << "width : " << shape.width << ", height : " << shape.height;
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SHAPE_H_
