#include <rtac_base/types/common.h>
#include <rtac_base/algorithm.h>

#include <iostream>
using namespace std;
using namespace rtac::types;
using namespace rtac::algorithm;
using namespace rtac::types::indexing;

int main()
{
    Vector3<float> v = Vector3<float>::Zero();
    v(1) = 1.0;
    v(2) = 1.0;
    auto vc = find_noncolinear(v);
    auto vo = find_orthogonal(v);

    cout << "v     : " << v.transpose()  << endl;
    cout << "vc    : " << vc.transpose() << endl;
    cout << "dot   : " << v.dot(vc) << endl;
    cout << "cross : " << v.cross(vc).transpose() << endl;
    cout << "vo    : " << vo.transpose() << endl;
    cout << "dot   : " << v.dot(vo) << endl;
    cout << "cross : " << v.cross(vo).transpose() << endl;
    cout << endl;

    Matrix3<float> m  = Matrix3<float>::Identity() + 0.1*Matrix3<float>::Random();
    Matrix3<float> mo = orthonormalized(m);
    cout << "Random :\n" << m << endl;
    cout << "Orthonormalized :\n" << mo << endl;
    cout << "Check ortho :\n" << mo*mo.transpose() << endl;
    cout << endl;

    Matrix3<float> m2  = Matrix3<float>::Identity();
    m2(0,0) = 1e-7;
    try {
        // must fail.
        Matrix3<float> mo2 = orthonormalized(m2);
    }
    catch(const std::runtime_error& e) {
        cout << "Conditioning test : Task failed successfully." << endl;
    }
    
    cout << to_degrees(1.5) << endl;
    cout << to_radians(180.0) << endl;
    float radians = 1.5;
    float degrees = 90.0;
    cout << to_degrees(radians) << endl;
    cout << to_radians(degrees) << endl;

    return 0;
}
