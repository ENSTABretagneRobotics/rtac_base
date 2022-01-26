#include <rtac_base/types/ArrayView.h>
using namespace rtac::types::array;

StackArray<float, 3, 3> test0_0()
{
    StackArray<float, 3, 3> a;

    a(0,0) = 0; a(0,1) = 1; a(0,2) = 2;
    a(1,0) = 3; a(1,1) = 4; a(1,2) = 5;
    a(2,0) = 6; a(2,1) = 7; a(2,2) = 8;

    return a;
}

void test0_1(float* data)
{
    FixedArrayView<float, 3, 3> a(data);

    a(0,0) = 0; a(0,1) = 1; a(0,2) = 2;
    a(1,0) = 3; a(1,1) = 4; a(1,2) = 5;
    a(2,0) = 6; a(2,1) = 7; a(2,2) = 8;
}
