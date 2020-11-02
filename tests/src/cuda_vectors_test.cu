#include <iostream>
using namespace std;

#include <rtac_base/cuda/SharedVectors.h>
using namespace rtac::cuda;

template <typename T> __global__
void add(T* data, T value)
{
    data[threadIdx.x] += value;
}


int main()
{
    HostVector<float> vh0(10);
    int n = 0;
    for(auto& value : vh0) {
        value = n;
        n++;
    }
    DeviceVector<float> vd0(vh0);

    cout << "vh0 : " << vh0 << endl;
    cout << "vd0 : " << vd0 << endl;

    
    add<float><<<1, vd0.size()>>>(vd0.data(), 10);
    cudaDeviceSynchronize();

    cout << "Modified vd0 :" << endl;
    cout << "vh0 : " << vh0 << endl;
    cout << "vd0 : " << vd0 << endl;

    return 0;
}


