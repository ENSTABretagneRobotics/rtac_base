#ifndef _DEF_RTAC_BASE_TYPES_ARRAY_SCALE_H_
#define _DEF_RTAC_BASE_TYPES_ARRAY_SCALE_H_

#include <tuple>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace types {

/**
 * Utility namespace for ArrayScale (mainly for a pack call on functors);
 */
namespace details {

/**
 * This get the return type of a functor.
 */
template <class FunctorT>
struct FunctorType {
    using type = typename std::invoke_result<
        decltype(&FunctorT::operator()), FunctorT, std::size_t>::type;
};

/**
 * This template type and its specialization check if the return types of a set
 * of functors return the same type.
 */
template <typename... FunctorTs> struct functors_compatible {};
template <typename FT0, typename FT1, typename... FTs>
struct functors_compatible<FT0, FT1, FTs...> {
    static constexpr bool value = 
        std::is_same<typename FunctorType<FT0>::type,
                     typename FunctorType<FT1>::type>::value &&
        functors_compatible<FT1, FTs...>::value;
};
template <typename FT0, typename FT1>
struct functors_compatible<FT0, FT1> {
    static constexpr bool value = 
        std::is_same<typename FunctorType<FT0>::type,
                     typename FunctorType<FT1>::type>::value;
};
template <typename FT0>
struct functors_compatible<FT0> {
    static constexpr bool value = true;
};


template <class TupleT, std::size_t Idx = std::tuple_size<TupleT>::value - 1>
struct functor_tuple_compatible {
    static constexpr bool value = functors_compatible<
        typename std::tuple_element<Idx,   TupleT>::type,
        typename std::tuple_element<Idx-1, TupleT>::type>::value &&
        functor_tuple_compatible<TupleT,Idx-1>::value;
};
template <class TupleT>
struct functor_tuple_compatible<TupleT, 0> {
    static constexpr bool value = true;
};

/**
 * Simple helper to check the functors return type.
 * 
 * We check if all the return types are the same then typedef only the first
 * functor return type.
 */
template <typename... FTs> struct functors_return_type {};
template <typename FT, typename... FTs>
struct functors_return_type<FT,FTs...>
{
    static_assert(functors_compatible<FT, FTs...>::value,
        "All functors do not have the same return type for ArrayScale");
    using type = typename FunctorType<FT>::type;
};
template <typename TupleT>
struct functor_tuple_return_type {
    static_assert(functor_tuple_compatible<TupleT>::value,
        "All functors do not have the same return type for ArrayScale");
    using type = typename FunctorType<typename std::tuple_element<0,TupleT>::type>::type;
};

/**
 * This function dispatch the tuple of indexes indexes on the tuple of functors
 * and returns all the results in a std::array.
 */
template <class TupleT, class TupleIdxT, std::size_t... I> RTAC_HOSTDEVICE
constexpr decltype(auto) do_call_functors(TupleT&& t, TupleIdxT&& indexes,
                                          std::index_sequence<I...>)
{
    return std::array{(std::get<I>(std::forward<TupleT>(t)).operator()(
            std::get<I>(std::forward<TupleIdxT>(indexes))
        ))...};
}
template <class TupleT, typename... Indexes> RTAC_HOSTDEVICE
constexpr decltype(auto) call_functors(TupleT&& t, const Indexes&... indexes)
{
    return do_call_functors(std::forward<TupleT>(t), std::tie(indexes...),
        std::make_index_sequence<sizeof...(Indexes)>{});
}

/**
 * This function selects a functor from a tuple at runtime and calls it.
 */
template <typename TupleT, std::size_t I = std::tuple_size<TupleT>::value>
struct runtime_select_functor {
    static RTAC_HOSTDEVICE typename functor_tuple_return_type<TupleT>::type 
    call(TupleT&& t, std::size_t elementIdx, std::size_t idx) {
        if(I-1 == elementIdx) {
            return std::get<I-1>(std::forward<TupleT>(t)).operator()(idx);
        }
        return runtime_select_functor<TupleT,I-1>::call(std::forward<TupleT>(t), elementIdx, idx);
    }
};
template <typename TupleT>
struct runtime_select_functor<TupleT,0> {
    static RTAC_HOSTDEVICE typename functor_tuple_return_type<TupleT>::type 
    call(TupleT&& t, std::size_t elementIdx, std::size_t idx) {
        #ifdef RTAC_KERNEL // can't throw exception in cuda device code
        assert(0); // Will reach here if user tries to access tuple element
                   // greater then size of tuple
        #else
        throw std::out_of_range("Invalid tuple element index");
        #endif
    }
};
template <class TupleT> RTAC_HOSTDEVICE
typename functor_tuple_return_type<TupleT>::type 
    call_one_functor(TupleT&& t, std::size_t functorIdx, std::size_t idx)
{
    return runtime_select_functor<TupleT>::call(std::forward<TupleT>(t), functorIdx, idx);
}

}; //namespace details

/**
 * The ScaleFunctor defines the interface of a single dimension scale functor,
 * i.e. the mapping between the indexes of an array and the corresponding scale
 * of the physical dimension.
 */
template <class Derived>
struct ScaleFunctor {
    /**
     * Mapping between array index and scale of the corresponding physical
     * dimension.
     */
    RTAC_HOSTDEVICE auto operator()(std::size_t idx) const {
        return (*reinterpret_cast<const Derived*>(this))(idx);
    }
};

template <typename... Ts>
class ArrayScale 
{
    public:

    using value_type = typename details::functors_return_type<Ts...>::type;
    using TupleType  = std::tuple<Ts...>;

    static constexpr std::size_t 
        Dimensionality = std::tuple_size<std::tuple<Ts...>>::value;
    using OutputType   = std::array<value_type, Dimensionality>;

    protected:

    std::tuple<Ts...> scales_;

    public:

    ArrayScale(const Ts&... scales) : scales_(scales...) {}

    /**
     * This calls operator() on each scale functor in scales_ and returns the
     * result as a std::array.
     */
    template <typename... Indexes> RTAC_HOSTDEVICE
    std::array<value_type, Dimensionality> operator()(Indexes... indexes) const {
        static_assert(sizeof...(Ts) == sizeof...(Indexes),
                      "Number of indices on ArrayScale call does not match Dimension count");
        return details::call_functors(std::forward<const TupleType>(scales_), indexes...);
    }

    RTAC_HOSTDEVICE value_type get(std::size_t dimIdx, std::size_t idx) const {
        return details::call_one_functor(std::forward<const TupleType>(scales_), dimIdx, idx);
    }

    template <std::size_t Idx>
    RTAC_HOSTDEVICE const auto& scale() const { 
        return std::get<Idx>(std::forward<const TupleType>(scales_));
    }
    template <std::size_t Idx>
    RTAC_HOSTDEVICE auto& scale() { 
        return std::get<Idx>(scales_);
    }
    
};

}; //namespace types
}; //namespace rtac


#endif //_DEF_RTAC_BASE_TYPES_ARRAY_SCALE_H_


