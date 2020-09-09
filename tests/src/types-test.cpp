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
    cout << endl;

    rtac::types::Matrix3<float> m = rtac::types::Matrix3<float>::Random();
    rtac::types::Pose<float> pose2(rtac::types::Vector3<float>(), m);
    cout << "Random :\n" << m << endl;
    cout << "Pose :\n" << pose2 << endl;
    cout << "Check :\n" << pose2.rotation_matrix()*pose2.rotation_matrix().transpose() << endl;

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
    cout << "Cube0 :\n" << cube << endl;

#ifdef RTAC_BASE_PLY_FILES
    cube.export_ply("cube.ply");

    auto cube1 = rtac::types::Mesh<>::from_ply("cube.ply");
    cout << "Cube1 :\n" << cube1 << endl;
#endif

    return 0;
}
