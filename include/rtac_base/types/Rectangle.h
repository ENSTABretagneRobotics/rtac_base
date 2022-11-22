#ifndef _DEF_RTAC_BASE_TYPES_RECTANGLE_H_
#define _DEF_RTAC_BASE_TYPES_RECTANGLE_H_

#include <iostream>

#include <rtac_base/types/Shape.h>

namespace rtac {

/**
 * Represent an Area Of Interest on a 2D surface (screen, image...)
 */
template <typename T>
class Rectangle
{
    public:
    
    T left;
    T right;
    T bottom;
    T top;

    T width() const;
    T height() const;
    Shape<T> shape() const;
};

template <typename T>
T Rectangle<T>::width() const
{
    return right - left;
}

template <typename T>
T Rectangle<T>::height() const
{
    return top - bottom;
}

template <typename T>
Shape<T> Rectangle<T>::shape() const
{
    return Shape<T>({this->width(), this->height()});
}

}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::Rectangle<T>& rect)
{
    os << "l,r : [" << rect.left << ", " << rect.right
       << "], b,t : [" << rect.bottom << ", " << rect.top << "]";
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_RECTANGLE_H_
