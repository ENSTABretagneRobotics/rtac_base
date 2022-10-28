#include <iostream>
using namespace std;

#include <rtac_base/types/Bounds.h>
using namespace rtac::types;

int main()
{
    auto if32 = Bounds<float>::oo();
    auto iu32 = Bounds<uint32_t>::oo();
    auto ii16 = Bounds<int16_t>::oo();

    cout << "Bounds<float>::oo()    : " << if32 << endl;
    cout << "Bounds<uint32_t>::oo() : " << iu32 << endl;
    cout << "Bounds<int16_t>::oo()  : " << ii16 << endl;
    cout << endl;

    auto if32_2 = Bounds<float,2>::oo();
    auto iu32_3 = Bounds<uint32_t,3>::oo();
    auto ii16_4 = Bounds<int16_t,4>::oo();

    cout << "Bounds<float,2>::oo()    :\n" << if32_2 << endl;
    cout << "Bounds<uint32_t,3>::oo() :\n" << iu32_3 << endl;
    cout << "Bounds<int16_t,4>::oo()  :\n" << ii16_4 << endl;
    cout << endl;

    auto zf32 = Bounds<float>::Zero();
    auto zu32 = Bounds<uint32_t>::Zero();
    auto zi16 = Bounds<int16_t>::Zero();

    cout << "Bounds<float>::Zero()    : " << zf32 << endl;
    cout << "Bounds<uint32_t>::Zero() : " << zu32 << endl;
    cout << "Bounds<int16_t>::Zero()  : " << zi16 << endl;
    cout << endl;

    auto zf32_2 = Bounds<float,2>::Zero();
    auto zu32_3 = Bounds<uint32_t,3>::Zero();
    auto zi16_4 = Bounds<int16_t,4>::Zero();

    cout << "Bounds<float,2>::Zero()    :\n" << zf32_2 << endl;
    cout << "Bounds<uint32_t,3>::Zero() :\n" << zu32_3 << endl;
    cout << "Bounds<int16_t,4>::Zero()  :\n" << zi16_4 << endl;
    cout << endl;

    zf32.update(10);
    zu32.update(10);
    zi16.update(10);

    cout << "Bounds<float>::Zero()    update(10) : " << zf32 << endl;
    cout << "Bounds<uint32_t>::Zero() update(10) : " << zu32 << endl;
    cout << "Bounds<int16_t>::Zero()  update(10) : " << zi16 << endl;
    cout << endl;
    
    return 0;
}
