#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>

#include <iostream>
using namespace std;
using namespace rtac::indexing;

int main()
{
    auto pose = rtac::Pose<float>::Identity();
    cout << pose << endl;
    cout << pose.rotation_matrix() << endl << endl;
    cout << pose.homogeneous_matrix() << endl << endl;
    cout << endl;
    
    rtac::Matrix3<float> m = rtac::Matrix3<float>::Random();
    auto pose2 = rtac::Pose<float>::from_rotation_matrix(m, rtac::Vector3<float>());
    pose2.normalize();
    cout << "Random :\n" << m << endl;
    cout << "Pose :\n" << pose2 << endl;
    cout << "Check :\n" << pose2.rotation_matrix()*pose2.rotation_matrix().transpose() << endl;

    cout << "Inverse : " << pose2.inverse() << endl;
    cout << "Product : " << pose2*pose2.inverse() << endl;

    rtac::Matrix<float> A(3,3);
    cout << A << endl << endl;
    A << 1,2,3,
         4,5,6,
         7,8,9;
    cout << A << endl << endl;

    cout << A(0,all) << endl << endl;
    cout << A(seq(0,1), all) << endl << endl;

    return 0;
}
