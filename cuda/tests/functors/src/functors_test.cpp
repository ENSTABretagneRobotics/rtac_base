#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include "functors_test.h"

int main()
{
    int N = 10;

    HostVector<float> input(N);
    for(int n = 0; n < N; n++) {
        input[n] = n;
    }

    //auto output = scaling(input, functor::Scaling<float>({2.0f}));

    auto f = Saxpy(functor::Offset<float>({3.0f}), functor::Scaling<float>({2.0f}));
    cout << f(1.0f) << endl;

    auto output = saxpy(input, Saxpy(functor::Offset<float>({3.0f}),
                                     functor::Scaling<float>({2.0f})));

    cout << input  << endl;
    cout << output << endl;

    return 0;
}
