#ifndef _DEF_RTAC_BASE_TYPES_TUPLE_POINTER_H_
#define _DEF_RTAC_BASE_TYPES_TUPLE_POINTER_H_

#include <tuple>
#include <type_traits>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

namespace details {

template <class TupleT, std::size_t Idx = std::tuple_size<TupleT>::value - 1>
struct tuple_do
{
    using element_type = typename std::tuple_element<Idx,TupleT>::type;

    RTAC_HOSTDEVICE static void increment(TupleT& t) {
        std::get<Idx>(t)++;
        tuple_do<TupleT,Idx-1>::increment(t);
    }
    RTAC_HOSTDEVICE static void decrement(TupleT& t) {
        std::get<Idx>(t)--;
        tuple_do<TupleT,Idx-1>::decrement(t);
    }

    RTAC_HOSTDEVICE static void upshift(TupleT& t, int shift) {
        std::get<Idx>(t) += shift;
        tuple_do<TupleT,Idx-1>::upshift(t, shift);
    }
    RTAC_HOSTDEVICE static void downshift(TupleT& t, int shift) {
        std::get<Idx>(t) -= shift;
        tuple_do<TupleT,Idx-1>::downshift(t, shift);
    }

    RTAC_HOSTDEVICE static bool not_equal(const TupleT& lhs, const TupleT& rhs) {
        if(std::get<Idx>(lhs) != std::get<Idx>(rhs))
            return true;
        else
            return tuple_do<TupleT,Idx-1>::not_equal(lhs,rhs);
    }
};

template <class TupleT>
struct tuple_do<TupleT,0>
{
    using element_type = typename std::tuple_element<0,TupleT>::type;

    RTAC_HOSTDEVICE static void increment(TupleT& t) { 
        std::get<0>(t)++;
    }
    RTAC_HOSTDEVICE static void decrement(TupleT& t) { 
        std::get<0>(t)--;
    }

    RTAC_HOSTDEVICE static void upshift(TupleT& t, int shift) {
        std::get<0>(t) += shift;
    }
    RTAC_HOSTDEVICE static void downshift(TupleT& t, int shift) {
        std::get<0>(t) -= shift;
    }


    RTAC_HOSTDEVICE static bool not_equal(const TupleT& lhs, const TupleT& rhs) {
        return (std::get<0>(lhs) != std::get<0>(rhs));
    }
};

template <typename TupleT>
struct IndexSequence {
    static constexpr const std::size_t Size =
        std::tuple_size<typename std::remove_reference<TupleT>::type>::value;
    using type = std::make_index_sequence<Size>;
};

template <class TupleT, std::size_t... I>
constexpr decltype(auto) do_dereference_pointer(TupleT&& t, std::index_sequence<I...>)
{
    return std::tie((*std::get<I>(std::forward<TupleT>(t)))...);
}
template <class TupleT>
constexpr decltype(auto) dereference_pointer(TupleT&& t) {
    return do_dereference_pointer(std::forward<TupleT>(t), 
        std::make_index_sequence<std::tuple_size<
            typename std::remove_reference<TupleT>::type>::value>{});
}

template <class TupleT, std::size_t... I>
constexpr decltype(auto) do_dereference_pointer_shift(TupleT&& t, 
                                                      int shift,
                                                      std::index_sequence<I...>)
{
    return std::tie((*(std::get<I>(std::forward<TupleT>(t))+shift))...);
}
template <class TupleT>
constexpr decltype(auto) dereference_pointer_shift(TupleT&& t, int shift) {
    return do_dereference_pointer_shift(std::forward<TupleT>(t), shift,
        std::make_index_sequence<std::tuple_size<
            typename std::remove_reference<TupleT>::type>::value>{});
}

/**
 * These make a std::tuple<const Ts*...> from a std::tuple<const Ts*...>
 *
 * Maybe
 */
template <typename... Ts> RTAC_HOSTDEVICE
inline std::tuple<const Ts*...> do_make_const_tuple(const Ts*... args)
{
    return std::make_tuple(args...);
}
template <class TupleT, std::size_t... I> RTAC_HOSTDEVICE
inline decltype(auto) make_const_tuple_stub(TupleT&& t,
                                               std::index_sequence<I...>)
{
    return do_make_const_tuple((std::get<I>(std::forward<TupleT>(t)))...);
}
template <class TupleT> RTAC_HOSTDEVICE
inline decltype(auto) make_const_tuple(TupleT&& t)
{
    return make_const_tuple_stub(std::forward<TupleT>(t),
        std::make_index_sequence<std::tuple_size<
            typename std::remove_reference<TupleT>::type>::value>{});
}

}; //namespace details

/**
 * This is an object holding a std::tuple of independent pointers.
 *
 * When dereferenced, returns a std::tuple of the underlying value types (or
 * references).
 */
template <typename... Ts>
struct TuplePointer
{
    using TupleType = std::tuple<Ts*...>;

    using value_type = std::tuple<Ts...>;
    using reference  = std::tuple<Ts&...>;

    static constexpr const std::size_t Size = std::tuple_size<TupleType>::value;

    TupleType data;

    RTAC_HOSTDEVICE TuplePointer& operator++()
    {
        details::tuple_do<TupleType>::increment(data);
        return *this;
    }
    RTAC_HOSTDEVICE TuplePointer operator++(int)
    {
        TuplePointer other(*this);
        ++(*this);
        return other;
    }
    RTAC_HOSTDEVICE TuplePointer& operator--()
    {
        details::tuple_do<TupleType>::decrement(data);
        return *this;
    }
    RTAC_HOSTDEVICE TuplePointer operator--(int)
    {
        TuplePointer other(*this);
        --(*this);
        return other;
    }

    RTAC_HOSTDEVICE TuplePointer& operator+=(int shift)
    {
        details::tuple_do<TupleType>::upshift(data, shift);
        return *this;
    }
    RTAC_HOSTDEVICE TuplePointer& operator-=(int shift)
    {
        details::tuple_do<TupleType>::downshift(data, shift);
        return *this;
    }

    RTAC_HOSTDEVICE bool operator!=(const TuplePointer& other) const
    {
        return details::tuple_do<TupleType>::not_equal(this->data, other.data);
    }
    RTAC_HOSTDEVICE bool operator==(const TuplePointer& other) const
    {
        return !(*this != other);
    }

    RTAC_HOSTDEVICE bool operator<(const TuplePointer& other) const
    {
        return std::get<0>(data) < std::get<0>(other.data);
    }
    RTAC_HOSTDEVICE bool operator<=(const TuplePointer& other) const
    {
        return std::get<0>(data) <= std::get<0>(other.data);
    }
    RTAC_HOSTDEVICE bool operator>(const TuplePointer& other) const
    {
        return std::get<0>(data) > std::get<0>(other.data);
    }
    RTAC_HOSTDEVICE bool operator>=(const TuplePointer& other) const
    {
        return std::get<0>(data) >= std::get<0>(other.data);
    }

    RTAC_HOSTDEVICE int distance_from(const TuplePointer& other) const
    {
        return std::get<0>(data) - std::get<0>(other.data);
    }

    RTAC_HOSTDEVICE value_type operator*() const
    {
        return details::dereference_pointer(data);
    }
    RTAC_HOSTDEVICE value_type operator[](int idx) const
    {
        return details::dereference_pointer_shift(data, idx);
    }

    RTAC_HOSTDEVICE reference operator*()
    {
        return details::dereference_pointer(data);
    }
    RTAC_HOSTDEVICE reference operator[](int idx)
    {
        return details::dereference_pointer_shift(data, idx);
    }
    
    RTAC_HOSTDEVICE TuplePointer<const Ts...> make_const() const
    {
        return TuplePointer<const Ts...>{details::make_const_tuple(data)};
    }
};

}; //namespace types
}; //namespace rtac

template <typename... Ts>
RTAC_HOSTDEVICE inline rtac::types::TuplePointer<Ts...> 
    operator+(const rtac::types::TuplePointer<Ts...>& p, int shift)
{
    rtac::types::TuplePointer<Ts...> res = p;
    res += shift;
    return res;
}

template <typename... Ts>
RTAC_HOSTDEVICE inline rtac::types::TuplePointer<Ts...> operator+(int shift, 
    const rtac::types::TuplePointer<Ts...>& p)
{
    return p + shift;
}

template <typename... Ts>
RTAC_HOSTDEVICE inline rtac::types::TuplePointer<Ts...> 
    operator-(const rtac::types::TuplePointer<Ts...>& p, int shift)
{
    rtac::types::TuplePointer<Ts...> res = p;
    res -= shift;
    return res;
}

template <typename... Ts>
RTAC_HOSTDEVICE inline int operator-(const rtac::types::TuplePointer<Ts...>& lhs, 
                     const rtac::types::TuplePointer<Ts...>& rhs)
{
    return lhs.distance_from(rhs);
}

#endif //_DEF_RTAC_BASE_TYPES_TUPLE_POINTER_H_


