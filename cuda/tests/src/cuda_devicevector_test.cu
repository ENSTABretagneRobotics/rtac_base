#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/cuda/CudaVector.h>
using CudaVector = rtac::cuda::CudaVector<float>;
using HostVector = rtac::HostVector<float>;

std::vector<float> new_vector(size_t size)
{
    std::vector<float> res(size);
    int i = 0;
    for(auto& value : res) {
        value = i; i++;
    }
    return res;
}

__global__ void add(float* p, size_t size)
{
    for(int i = 0; i < size; i++) {
        p[i] += 100;
    }
}

int main()
{
    int N = 10;
    auto v0 = new_vector(N);

    HostVector vh0(v0);
    CudaVector vd0(v0);
    HostVector vh1(vd0);
    CudaVector vd1(vh0);

    cout << "vh0 : " << vh0.data() << endl;
    cout << "vd0 : " << vd0.data() << endl;
    cout << "vh1 : " << vh1.data() << endl;
    cout << "vd1 : " << vd1.data() << endl;

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

    add<<<1,1>>>(vd0.data(), vd0.size());
    vd1 = vd0;

    cout << "vd0 : " << vd0 << endl;
    cout << "vd1 : " << vd1 << endl;

    cout << "vh0 : " << vh0.data() << endl;
    cout << "vd0 : " << vd0.data() << endl;
    cout << "vh1 : " << vh1.data() << endl;
    cout << "vd1 : " << vd1.data() << endl;

    return 0;
}
