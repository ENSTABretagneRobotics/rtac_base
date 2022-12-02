#include <iostream>
using namespace std;

#include <rtac_base/geometry.h>
#include <rtac_base/cuda/geometry.h>
using namespace rtac;

int main()
{
    auto m0 = Eigen::Matrix3<float>::Identity();
    Eigen::Vector3<float> v0(1,2,3);

    //cout << vector_get(m0, 0) << endl; // fails successfully at compile time
    cout << vector_get(v0, 0) << endl;

    float3 f3 = make_float3(v0);
    cout << f3 << endl;

    float4 f4 = make_float4(v0); // does not fail at compile time
    cout << f4 << endl;

    return 0;
}
