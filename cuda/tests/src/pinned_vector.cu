#include <iostream>
using namespace std;

#include <rtac_base/cuda/PinnedVector.h>
#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;
using namespace rtac;

template <typename T>
__global__ void fill_vector(rtac::VectorView<T> vect, T offset = 0)
{
    for(auto idx = threadIdx.x; idx < vect.size(); idx += blockDim.x) {
        vect[idx] = idx + offset;
    }
}

int main()
{
    PinnedVector<int> p0(10);
    fill_vector<<<1,512>>>(p0.view());
    cout << p0 << endl;

    DeviceVector<int> d0(10);
    fill_vector<<<1,512>>>(d0.view(),10);
    p0 = d0;
    cout << p0 << endl;


    fill_vector<<<1,512>>>(d0.view(),20);
    HostVector<int> h0(d0);
    p0 = h0;
    cout << p0 << endl;

    fill_vector<<<1,512>>>(p0.view(),30);
    h0 = p0;
    d0 = p0;
    cout << h0 << endl;
    cout << d0 << endl;

    return 0;
}


