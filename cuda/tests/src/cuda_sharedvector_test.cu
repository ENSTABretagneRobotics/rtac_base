#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using DeviceVector = rtac::cuda::SharedDeviceVector<float>;
using HostVector   = rtac::cuda::SharedHostVector<float>;

DeviceVector new_vector(size_t size)
{
    HostVector res(size);
    for(int i = 0; i < size; i++) {
        res.data()[i] = i;
    }
    return res;
}

__global__ 
void add(float* p, size_t size)
{
    for(int i = 0; i < size; i++) {
        p[i] += 10;
    }
}

int main()
{
    DeviceVector vd0(new_vector(10));
    DeviceVector vd1(vd0);
    DeviceVector vd2(vd0.copy());
    
    cout << "vd0 : " << vd0 << endl;
    cout << "vd1 : " << vd1 << endl;
    cout << "vd2 : " << vd2 << endl;
    cout << "Modifying vd0 :" << endl;
    add<<<1,1>>>(vd0.data(), vd0.size());
    cudaDeviceSynchronize();
    cout << "vd0 : " << vd0 << endl;
    cout << "vd1 (shallow) : " << vd1 << endl;
    cout << "vd2 (deep)    : " << vd2 << endl;

    return 0;
}

