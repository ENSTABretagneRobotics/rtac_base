#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Mesh.h>

#include <iostream>
using namespace std;
using namespace rtac::types::indexing;

int main()
{
    dummy();
    rtac::types::Pose<float> pose;
    cout << pose << endl;
    cout << pose.rotation_matrix() << endl << endl;
    cout << pose.homogeneous_matrix() << endl << endl;

    rtac::types::Matrixf A(3,3);
    cout << A << endl << endl;
    A << 1,2,3,
         4,5,6,
         7,8,9;
    cout << A << endl << endl;

    cout << A(0,all) << endl << endl;
    cout << A(seq(0,1), all) << endl << endl;

    rtac::types::Array3f B(3);
    cout << B << endl;

    auto cube = rtac::types::Mesh<>::cube();
    cout << cube << endl;

#ifdef RTAC_BASE_PLY_FILES
    cube.export_ply("cube.ply");

    auto cube1 = rtac::types::Mesh<>::from_ply("cube.ply");
    cout << cube1 << endl;
#endif

    return 0;
}
