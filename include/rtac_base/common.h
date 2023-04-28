#ifndef _DEF_RTAC_BASE_COMMON_H_
#define _DEF_RTAC_BASE_COMMON_H_

#if defined(__clang__)
    #define RTAC_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
    #define RTAC_GCC
#elif defined(_MSC_VER)
  #define RTAC_MSVC
#endif

#if !defined(RTAC_GCC) && !defined(RTAC_CLANG) && !defined(RTAC_MSVC)
    #error "Unsupported compiler. Should be GCC, CLANG or MSVC"
#endif

#define RTAC_PACKED_STRUCT(...) \
    _Pragma("pack(push,1)")     \
    __VA_ARGS__                 \
    _Pragma("pack(pop)")

#include <rtac_base/cuda_defines.h>

#endif //_DEF_RTAC_BASE_COMMON_H_


