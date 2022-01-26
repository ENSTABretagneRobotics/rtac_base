#include <array>

std::array<float,9> test1_0()
{
    std::array<float,9> a;
    a[0] = 0; a[1] = 1; a[2] = 2;
    a[3] = 3; a[4] = 4; a[5] = 5;
    a[6] = 6; a[7] = 7; a[8] = 8;

    return a;
}

void test1_1(float* a)
{
    a[0] = 0; a[1] = 1; a[2] = 2;
    a[3] = 3; a[4] = 4; a[5] = 5;
    a[6] = 6; a[7] = 7; a[8] = 8;
}
