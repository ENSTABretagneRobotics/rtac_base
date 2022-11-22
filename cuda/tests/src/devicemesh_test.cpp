#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceMesh.h>
using namespace rtac::cuda;

#include <rtac_base/types/Mesh.h>
using namespace rtac;

int main()
{
    auto mesh = DeviceMesh<>::cube();
    cout << *mesh << endl;

    HostMesh<>(*mesh).export_ply("device_cube.ply", true);

    auto loaded = DeviceMesh<>::from_ply("device_cube.ply");
    cout << "Loaded " << *loaded << endl;
    
    auto cube0 = HostMesh<>::cube();

    DeviceMesh<> copied;
    copied = *cube0;
    cout << "Device copied " << copied << endl;

    HostMesh<> hostCopied;
    hostCopied = copied;
    cout << "Host copied " << hostCopied << endl;

    return 0;
}
