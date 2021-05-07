#ifndef _DEF_RTAC_BASE_TYPE_UTILS_H_
#define _DEF_RTAC_BASE_TYPE_UTILS_H_

#include <cstring>
#include <type_traits>
#include <tuple>

namespace rtac { namespace types {

// TupleTypeIndex ///////////////////////////////////////////////////////////
// The following structs will calculate at compile time the first index of the
// type "T" in std::tuple<Types...> (outputs a compile time constant).  If T is
// not in Types, the total number of Types in std::tuple<Types...> is returned
// (which also is a compile time constant).
// For example : 
//     TupleTypeIndex<int,   std::tuple<float, int, double>>::value == 1;
//     TupleTypeIndex<float, std::tuple<float, int, double>>::value == 0;
//     TupleTypeIndex<char,  std::tuple<float, int, double>>::value == 3;

template <class T, class Tuple>
struct TupleTypeIndex;

// The index is calculated by successively specializing TupleTypeIndex.
// If we get the following template specialization, we found T in Types.
template <class T, class... Types>
struct TupleTypeIndex<T, std::tuple<T, Types...>> {
    static constexpr unsigned int value = 0;
};

// If we get the following template specialization, we did not found T, but we
// have tp stop the iteration anyway. Final index will be equal to original
// sizeof...(Types).
template <class T>
struct TupleTypeIndex<T, std::tuple<>> {
    static constexpr unsigned int value = 0;
};

// If we get the following template specialization, U is not T, so we continue
// iterating and we increment value.
template <class T, class U, class... Types>
struct TupleTypeIndex<T, std::tuple<U, Types...>> {
    static constexpr unsigned int value = 1 + TupleTypeIndex<T, std::tuple<Types...>>::value;
};
// End of TupleTypeIndex ///////////////////////////////////////////////////////////

// TupleTypeIndex ///////////////////////////////////////////////////////////
// The following struct will test at compile time if type "T" is part of the
// Tuple type types.
template <class T, class Tuple>
struct TypeInTuple : std::conditional<
    // Checking the index of Type in Tuple. If it returns the size of Tuple,
    // that means T is not part of Tuple.
    TupleTypeIndex<T,Tuple>::value == std::tuple_size<Tuple>::value,
    std::false_type, std::true_type>::type
{};

template <typename T>
inline T zero()
{
    // Simple one-liner to create an initialize an instance of type T.
    // (helpfull when working with pure C library)
    T res;
    std::memset(&res, 0, sizeof(T));
    return res;
}

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPE_UTILS_H_
