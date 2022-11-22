#ifndef _DEF_RTAC_BASE_TYPE_UTILS_H_
#define _DEF_RTAC_BASE_TYPE_UTILS_H_

#include <cstring>

namespace rtac {

template <typename T>
inline T zero()
{
    // Simple one-liner to create an initialize an instance of type T.
    // (helpfull when working with pure C library)
    T res;
    std::memset(&res, 0, sizeof(T));
    return res;
}

}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPE_UTILS_H_


