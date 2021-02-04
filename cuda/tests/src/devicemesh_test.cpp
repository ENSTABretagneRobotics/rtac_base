#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceMesh.h>
using namespace rtac::cuda;

#include <rtac_base/types/Mesh.h>
using namespace rtac::types;

int main()
{
    auto mesh = DeviceMesh<>::cube();
    cout << mesh << endl;

    mesh.export_ply("device_cube.ply", true);

    auto loaded = DeviceMesh<>::from_ply("device_cube.ply");
    cout << "Loaded " << loaded << endl;
    
    auto cube0 = Mesh<DeviceMesh<>::Point, DeviceMesh<>::Face, HostVector>::cube();

    DeviceMesh<> copied;
    copied = cube0;
    cout << "Device copied " << copied << endl;

    auto hostCopied = Mesh<DeviceMesh<>::Point, DeviceMesh<>::Face, HostVector>();
    hostCopied = cube0;
    cout << "Host copied " << hostCopied << endl;

    return 0;
}
