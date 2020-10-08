#include <iostream>
using namespace std;

#include <rtac_base/types/PointCloud.h>
using namespace rtac::types;

void print_ref(const PointCloud<>::PointCloudType& pc)
{
    cout << "Implicit reference cast" << endl;
    cout << pc;
}

void print_ptr(const PointCloud<>::ConstPtr& pc)
{
    cout << "Implicit pointer cast" << endl;
    cout << *pc;
}

int main()
{
    PointCloud<> pc(8);

    pc[0] = Point3D({0,0,0});
    pc[1] = Point3D({0,0,1});
    pc[2] = Point3D({0,1,0});
    pc[3] = Point3D({0,1,1});
    pc[4] = Point3D({1,0,0});
    pc[5] = Point3D({1,0,1});
    pc[6] = Point3D({1,1,0});
    pc[7] = Point3D({1,1,1});
    
    cout << pc;

    print_ref(pc);
    print_ptr(pc);

    return 0;
}
