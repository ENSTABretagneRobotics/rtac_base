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
    return 0;
}
