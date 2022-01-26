#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/ArrayView.h>
using namespace rtac::types::array;

int main()
{
    std::vector<int> data({1,2,3,4});

    VectorView<int> v0(data.data());
    cout << v0[0] << " " << v0[1] << " " << v0[2] << " " << v0[3] << endl;
    v0[0] = 5; v0[1] = 6; v0[2] = 7; v0[3] = 8;

    VectorView<const int> v1(data.data());
    cout << v1[0] << " " << v1[1] << " " << v1[2] << " " << v1[3] << endl;
    //v1[0] = 5; v1[1] = 6; v1[2] = 7; v1[3] = 8;

    StackVector<int,4> v2;
    v2[0] = 0; v2[1] = 1; v2[2] = 2; v2[3] = 3;
    cout << v2[0] << " " << v2[1] << " " << v2[2] << " " << v2[3] << endl;

    FixedArrayView<const int, 2, 2> a0(v2.data());
    cout << a0(0,0) << " " << a0(0,1) << " " << a0(1,0) << " " << a0(1,1) << endl;
    //a0(0,0) = 4; a0(0,1) = 5; a0(1,0) = 6; a0(1,1) = 7; // does not compile (as expected)

    FixedArrayView<int, 2, 2> a1(v2.data());
    a1(0,0) = 4; a1(0,1) = 5; a1(1,0) = 6; a1(1,1) = 7;
    cout << a0(0,0) << " " << a0(0,1) << " " << a0(1,0) << " " << a0(1,1) << endl;

    ArrayView<int> a2(2,2,v2.data());
    cout << a2(0,0) << " " << a2(0,1) << " " << a2(1,0) << " " << a2(1,1) << endl;
    a1(0,0) = 8; a1(0,1) = 9; a1(1,0) = 10; a1(1,1) = 11;
    cout << a0(0,0) << " " << a0(0,1) << " " << a0(1,0) << " " << a0(1,1) << endl;

    return 0;
}
