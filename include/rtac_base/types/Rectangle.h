#ifndef _DEF_RTAC_BASE_TYPES_RECTANGLE_H_
#define _DEF_RTAC_BASE_TYPES_RECTANGLE_H_

#include <iostream>

namespace rtac { namespace types {

template <typename T>
class Rectangle
{
    public:
    
    T left;
    T right;
    T bottom;
    T top;

    T xspan() const;
    T yspan() const;
};

template <typename T>
T Rectangle<T>::xspan() const
{
    return right - left;
}

template <typename T>
T Rectangle<T>::yspan() const
{
    return top - bottom;
}

}; //namespace types
}; //namespace rtac

template <typename T>
std::ostream& operator<<(std::ostream& os, const rtac::types::Rectangle<T>& rect)
{
    os << "l-r : [" << rect.left << "-" << rect.right
       << "], b-t : [" << rect.bottom << "-" << rect.top << "]";
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_RECTANGLE_H_
