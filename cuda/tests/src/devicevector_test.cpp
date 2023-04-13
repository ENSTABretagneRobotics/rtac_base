#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
using HostVector = rtac::HostVector<float>;
#include <rtac_base/cuda/CudaVector.h>
using CudaVector = rtac::cuda::CudaVector<float>;

std::vector<float> new_vector(size_t size)
{
    std::vector<float> res(size);
    int i = 0;
    for(auto& value : res) {
        value = i; i++;
    }
    return res;
}

int main()
{
    int N = 10;
    auto v0 = new_vector(N);

    HostVector vh0(v0);
    CudaVector vd0(v0);
    HostVector vh1(vd0);
    CudaVector vd1(vh0);

    cout << "vh0 : " << vh0 << endl;
    cout << "vd0 : " << vd0 << endl;
    cout << "vh1 : " << vh1 << endl;
    cout << "vd1 : " << vd1 << endl;

    vh0.data()[0] = 10;
    vd0 = vh0;
    vh1 = vd0;
    vd1 = vd0;
    vh1.data()[N-1] = 0;
    vh0 = vh1;
    cout << "vh0 : " << vh0 << endl;
    cout << "vd0 : " << vd0 << endl;
    cout << "vh1 : " << vh1 << endl;
    cout << "vd1 : " << vd1 << endl;

    return 0;
}
