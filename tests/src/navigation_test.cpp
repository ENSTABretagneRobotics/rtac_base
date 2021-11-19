#include <iostream>
using namespace std;

#include <rtac_base/navigation.h>
using namespace rtac::types;
using namespace rtac::navigation;

using Mat3 = Matrix3<double>;
using Vec3 = Vector3<double>;

int main()
{
    Mat3 r0ned = quaternion_from_nautical_degrees<double>(45,10,0).toRotationMatrix();
    Mat3 r0enu = ned_to_enu(r0ned);
    cout << r0ned << endl << endl;
    cout << r0enu << endl << endl;
    return 0;
}
