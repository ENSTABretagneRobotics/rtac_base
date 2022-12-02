#include <iostream>
using namespace std;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/PODWrapper.h>
using Pose = rtac::Pose<float>;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/DeviceReference.h>
using namespace rtac::cuda;

extern "C" {

    __constant__ rtac::PODWrapper<Pose> globalPose;
    //__constant__ Pose globalPose; // not compiling (expected not to)

}

__global__ void get_pose(Pose* out, Pose::Vec3* vout)
{
    *out  = *globalPose;
    *vout = globalPose->translation();
}

int main()
{
    auto p0 = Pose::from_translation(Pose::Vec3(1,2,3));
    cout << "p0 : " << p0 << endl;

    cudaMemcpyToSymbol(globalPose, &p0, sizeof(Pose));
    
    Pose       p1;
    Pose::Vec3 v1;

    get_pose<<<1,1>>>(Ref<Pose>(p1), Ref<Pose::Vec3>(v1));
    cudaDeviceSynchronize();

    cout << "p1 : " << p1             << endl;
    cout << "v1 : " << v1.transpose() << endl;

    return 0;
}

