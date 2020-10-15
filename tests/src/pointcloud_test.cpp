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
    PointCloud<> pc0(8);

    pc0[0] = Point3D({0,0,0});
    pc0[1] = Point3D({0,0,1});
    pc0[2] = Point3D({0,1,0});
    pc0[3] = Point3D({0,1,1});
    pc0[4] = Point3D({1,0,0});
    pc0[5] = Point3D({1,0,1});
    pc0[6] = Point3D({1,1,0});
    pc0[7] = Point3D({1,1,1});
    
    cout << pc0;
    pc0.export_ply("out.ply", false);
    auto reloaded = PointCloud<>::from_ply("out.ply");
    cout << "Reloaded :\n" << reloaded << reloaded.pose() << endl;

    print_ref(pc0);
    print_ptr(pc0);

    cout << "Deep copy" << endl;
    auto pc1 = pc0.copy();
    pc1[0].x = 10;
    cout << pc0;
    cout << pc1;

    cout << "Shallow copy" << endl;
    auto pc2 = pc0;
    pc2[0].x = 10;
    cout << pc0;
    cout << pc2;
    
    cout << pc0.pose() << endl;
    pc0.set_pose(Pose<float>({1,2,3}, {0,1,0,0}));
    cout << pc0.pose() << endl;

    return 0;
}
