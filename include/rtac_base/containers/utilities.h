#ifndef _DEF_RTAC_BASE_CONTAINERS_UTILITIES_H_
#define _DEF_RTAC_BASE_CONTAINERS_UTILITIES_H_

#include <type_traits>
#include <vector>

#include <rtac_base/containers/VectorView.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/Image.h>
#include <rtac_base/containers/ScaledArray.h>

namespace rtac {

//namespace details {
//
//template <class Container>
//struct ViewMaker {
//    static_assert(std::has_member_function_pointer<decltype(&Container::make_view)>::value,
//        "Container has no 'make_view' method. You should override the 'make_view' free function");
//    static auto make(const Container& container) { return container.view(); }
//    static auto make(Container& container)       { return container.view(); }
//};
//
//}

/**
 * Create a const view on a Container by calling the view method of the
 * container.
 *
 * User can overload this function to create custom view on types which do not
 * have a view method (such as std::vector)
 */
template <class Container>
auto make_view(const Container& container) {
    //return details::ViewMaker<Container>::make(container);
    return container.view();
}
template <class Container>
auto make_view(Container& container) {
    //return details::ViewMaker<Container>::make(container);
    return container.view();
}

/**
 * Creates a VectorView from a std::vector.
 */
template <typename T>
auto make_view(const std::vector<T>& container) {
    return VectorView<const T>(container.size(), container.data());
}

template <typename T>
auto make_view(std::vector<T>& container) {
    return VectorView<T>(container.size(), container.data());
}

}

#endif //_DEF_RTAC_BASE_CONTAINERS_UTILITIES_H_
