#ifndef _DEF_RTAC_BASE_TYPES_NDPOINT_H_
#define _DEF_RTAC_BASE_TYPES_NDPOINT_H_

#include <iostream>
#include <array>
#include <tuple>

namespace rtac {

namespace details {

template <typename... Ts> void dummy(Ts... args) {}

template <typename T0, typename T1, std::size_t S, std::size_t... I>
void do_assign_add_arrays(std::array<T0,S>& lhs, const std::array<T1,S>& rhs,
                          std::index_sequence<I...>)
{
    dummy((std::get<I>(lhs) += std::get<I>(rhs))...);
}

template <typename T0, typename T1, std::size_t S>
void assign_add_arrays(std::array<T0,S>& lhs, const std::array<T1,S>& rhs)
{
    do_assign_add_arrays(lhs, rhs, std::make_index_sequence<S>{});
}


};

template <typename T, std::size_t SizeV>
struct NDPoint : public std::array<T,SizeV>
{
    static constexpr std::size_t Size = SizeV;
    using value_type = T;

    template <typename Tother>
    NDPoint<T,Size> operator+=(const NDPoint<Tother,Size>& other)
    {
        details::assign_add_arrays(*this, other);
        return *this;
    }
};

}; //namespace rtac

template <typename T, std::size_t S>
inline std::ostream& operator<<(std::ostream& os, const rtac::NDPoint<T,S>& p)
{
    for(int i = 0; i < p.size(); i++) {
        os << " " << p[i];
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_NDPOINT_H_
