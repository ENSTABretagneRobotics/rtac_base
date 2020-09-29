#include <iostream>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/types/Mesh.h>
using namespace rtac;
using namespace rtac::types;

int main()
{
    auto path = files::find_one(".*lidar.ply");
    cout << "PLY file : " << path << endl;

    auto mesh = Mesh<float,uint32_t,3>::from_ply(path);

    cout << "Mesh loaded successfully" << endl;
    return 0;
}
