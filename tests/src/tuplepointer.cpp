#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/TuplePointer.h>
using namespace rtac::types;

template <typename... Ts>
inline std::ostream& operator<<(std::ostream& os, const TuplePointer<Ts...>& p);
template <typename... Ts>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& v);

template <typename T>
std::vector<T> generate_data(int N, int offset = 0)
{
    std::vector<T> res(N);
    for(int n = 0; n < N; n++) {
        res[n] = n + offset;
    }
    return res;
}

int main()
{
    int N = 10;
    auto fd = generate_data<float>(N);
    auto cd = generate_data<uint16_t>(N, 10);
    auto id = generate_data<size_t>(N, 100);

    TuplePointer<float, uint16_t, size_t> p0,p1;
    p0.data = std::make_tuple(fd.data(), cd.data(), id.data());

    cout << "p1 : " << p1 << endl;
    p1 = p0;
    cout << "p1 : " << p1 << endl;

    cout << "p0 : " << p0 << endl;
    cout << "p1 == p0 : " << (p1 == p0) << endl;
    ++p0;
    cout << "p0 : " << p0 << endl;
    p0++;
    cout << "p0 : " << p0 << endl;
    cout << "p1 == p0 : " << (p1 == p0) << endl;
    p0--; p0--;
    cout << "p0 : " << p0 << endl;
    cout << "p1 : " << p1 << endl;
    cout << "p1 == p0 : " << (p1 == p0) << endl;

    p1 += 16;
    cout << "p1 : " << p1 << endl;
    cout << "p1 == p0 : " << (p1 == p0) << endl;
    p1 -= 16;
    cout << "p1 : " << p1 << endl;
    cout << "p0 : " << p0 << endl;
    cout << "p1 == p0 : " << (p1 == p0) << endl;

    auto p2 = p0 + 8;
    cout << "p2 : " << p2 << endl;
    cout << "p0 : " << p0 << endl;
    cout << "p2 == p0 : " << (p2 == p0) << endl;
    p2 = p2 - 8;
    cout << "p2 : " << p2 << endl;
    cout << "p0 : " << p0 << endl;
    cout << "p2 == p0 : " << (p2 == p0) << endl;

    p2 += 16;
    cout << "p2 - p0  : " << p2 - p0 << endl;
    cout << "p2 > p0  : " << (p2 > p0) << endl;
    cout << "p2 < p0  : " << (p2 < p0) << endl;
    cout << "p2 >= p0 : " << (p2 >= p0) << endl;
    cout << "p2 <= p0 : " << (p2 <= p0) << endl;
    p2 = p0;
    cout << "p2 > p0  : " << (p2 > p0) << endl;
    cout << "p2 < p0  : " << (p2 < p0) << endl;
    cout << "p2 >= p0 : " << (p2 >= p0) << endl;
    cout << "p2 <= p0 : " << (p2 <= p0) << endl;
    p2--;
    cout << "p2 > p0  : " << (p2 > p0) << endl;
    cout << "p2 < p0  : " << (p2 < p0) << endl;
    cout << "p2 >= p0 : " << (p2 >= p0) << endl;
    cout << "p2 <= p0 : " << (p2 <= p0) << endl;

    std::get<0>(p0[9]) = 0.0f;
    std::get<1>(p0[9]) = 0.0f;
    std::get<2>(p0[9]) = 0.0f;

    for(int n = 0; n < N; n++) {
        cout << "p0 + " << n << " : " <<  *(p0 + n) << endl;
        cout << "p0[" << n << "]  : " <<  p0[n] << endl;
    }

    return 0;
}

template <class TupleT, std::size_t Idx = std::tuple_size<TupleT>::value - 1>
struct print_pointer_tuple {
    static void do_print(std::ostream& os, const TupleT& t)
    {
        print_pointer_tuple<TupleT,Idx-1>::do_print(os, t);
        os << " " << (const void*)std::get<Idx>(t);
    }
};
template <class TupleT>
struct print_pointer_tuple<TupleT,0> {
    static void do_print(std::ostream& os, const TupleT& t)
    {
        os << (const void*)std::get<0>(t);
    }
};

template <typename... Ts>
inline std::ostream& operator<<(std::ostream& os, const TuplePointer<Ts...>& p)
{
    print_pointer_tuple<typename TuplePointer<Ts...>::TupleType>::do_print(os, p.data);
    return os;
}

template <class TupleT, std::size_t Idx = std::tuple_size<TupleT>::value - 1>
struct print_tuple_value {
    static void do_print(std::ostream& os, const TupleT& t)
    {
        print_tuple_value<TupleT,Idx-1>::do_print(os, t);
        os << " " << std::get<Idx>(t);
    }
};
template <class TupleT>
struct print_tuple_value<TupleT,0> {
    static void do_print(std::ostream& os, const TupleT& t)
    {
        os << std::get<0>(t);
    }
};

template <typename... Ts>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& v)
{
    print_tuple_value<std::tuple<Ts...>>::do_print(os, v);
    return os;
}



