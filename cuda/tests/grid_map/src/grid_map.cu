#include <iostream>
using namespace std;

#include <rtac_base/types/GridMap.h>
using namespace rtac::types;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include "grid_map.hcu"

template <typename T>
class Affine
{
    public:

    using value_type = T;

    protected:

    T a_;
    T b_;

    public:

    Affine(T a, T b) : a_(a), b_(b) {}
    static Affine<T> Create(T lowerBound, T upperBound, std::size_t size) {
        auto extent = upperBound - lowerBound;
        return Affine<T>(extent / size, lowerBound + extent / (2*size));
    }

    RTAC_HOSTDEVICE T operator()(std::size_t idx) const { return a_*idx + b_; }
};

int main()
{
    int N = 10;

    auto affine = Affine<float>::Create(0.0f, 1.0f, N);
    for(int n = 0; n < N; n++) {
        cout << " " << affine(n);
    }
    cout << endl;

    GridMap gridMap(Affine<float>::Create(0.0f, 1.0f, N),
                     Affine<float>::Create(0.0f, 2.0f, N));
    for(int n = 0; n < N; n++) {
        cout << gridMap.map<0>()(n) << " "
             << gridMap.map<1>()(n) << endl;
    }
    
    cout << "Using GridMap :\n";
    for(int n = 0; n < N; n++) {
        auto res = gridMap(n,n);
        cout << res[0] << " " << res[1] << endl;
    }

    DeviceVector<float> mappingData(N);
    get_mapping<<<1, N>>>(mappingData.data(), GridMap(Affine<float>::Create(0.0f, 3.0f, N)));
    cudaDeviceSynchronize();

    cout << "mappingData : " << mappingData << endl;

    return 0;
}
