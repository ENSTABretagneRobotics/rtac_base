#ifndef _DEF_RTAC_BASE_TYPES_HANDLE_H_
#define _DEF_RTAC_BASE_TYPES_HANDLE_H_

#include <memory>

namespace rtac { namespace types {

// This is defined here to be able to change it depending on target system (or
// falling back on boost library if needed).
template <typename T>
using Handle     = std::shared_ptr<T>;
template <typename T>
using WeakHandle = std::weak_ptr<T>;

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_HANDLE_H_

