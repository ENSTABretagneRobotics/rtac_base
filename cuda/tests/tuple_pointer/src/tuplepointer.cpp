#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/time.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include <rtac_base/types/TuplePointer.h>
using namespace rtac::types;

#include "tuplepointer.h"

template <typename T>
T make_data(int i)
{
    return i;
}

template <>
Data make_data<Data>(int n)
{
    Data out;
    out.f1 = n;
    out.f2 = n + 10;
    out.f3 = n + 100;
    return out;
}

template <>
Data2 make_data<Data2>(int n)
{
    Data2 out;
    out.f32[0] = n;
    out.f32[1] = n + 10;
    out.f32[2] = n + 100;
    out.f32[3] = n + 1000;
    return out;
}

template <typename T>
DeviceVector<T> generate_data(int N, int offset = 0)
{
    std::vector<T> res(N);
    for(int n = 0; n < N; n++) {
        res[n] = make_data<T>(n) + offset;
    }
    return res;
}

int main()
{
    int N = 10000000;
    int M = 1000;

    // int N = 1000000;
    // int M = 10000;

    // int N = 100000;
    // int M = 100000;

    auto f32 = generate_data<float>(N);
    auto u16 = generate_data<uint16_t>(N, 10);
    auto u64 = generate_data<size_t>(N, 100);
    auto s   = generate_data<Data>(N);
    auto s2  = generate_data<Data2>(N);

    TuplePointer<float, uint16_t, size_t> p0;
    p0.data = std::make_tuple(f32.data(), u16.data(), u64.data());

    cout << "f32 : " << f32 << endl;
    cout << "u16 : " << u16 << endl;
    cout << "u64 : " << u64 << endl;

    do_stuff(p0, N);

    cout << "f32 : " << f32 << endl;
    cout << "u16 : " << u16 << endl;
    cout << "u64 : " << u64 << endl;
    
    rtac::time::Clock clock;
    for(int m = 0; m < M; m++)
        do_stuff(p0, N);
    auto t0 = clock.now<double>();
    
    clock.reset();
    for(int m = 0; m < M; m++)
        do_stuff_array(f32.data(), u16.data(), u64.data(), N);
    auto t1 = clock.now<double>();

    clock.reset();
    for(int m = 0; m < M; m++)
        do_stuff_struct(s.data(), N);
    auto t2 = clock.now<double>();

    cout << "TuplePointer    : " << 1000.0*t0 / M << "ms" << endl;
    cout << "Raw pointers    : " << 1000.0*t1 / M << "ms" << endl;
    cout << "Array of struct : " << 1000.0*t2 / M << "ms" << endl;

    auto f32_0 = generate_data<double>(N, 10);
    auto f32_1 = generate_data<double>(N, 10);
    auto f32_2 = generate_data<double>(N, 100);
    auto f32_3 = generate_data<double>(N, 1000);

    TuplePointer<double, double, double, double> p1;
    p1.data = std::make_tuple(f32_0.data(),
                              f32_1.data(),
                              f32_2.data(),
                              f32_3.data());
    clock.reset();
    for(int m = 0; m < M; m++)
        do_stuff(p1, N);
    auto t3 = clock.now<double>();
    
    clock.reset();
    for(int m = 0; m < M; m++)
        do_stuff_array(f32_0.data(),
                       f32_1.data(),
                       f32_2.data(),
                       f32_3.data(), N);
    auto t4 = clock.now<double>();

    clock.reset();
    for(int m = 0; m < M; m++)
        do_stuff_struct(s2.data(), N);
    auto t5 = clock.now<double>();

    cout << "Aligned TuplePointer    : " << 1000.0*t3 / M << "ms" << endl;
    cout << "Aligned raw pointers    : " << 1000.0*t4 / M << "ms" << endl;
    cout << "Aligned array of struct : " << 1000.0*t5 / M << "ms" << endl;

    return 0;
}
