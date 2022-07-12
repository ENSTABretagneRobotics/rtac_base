#include <iostream>
using namespace std;

#include <rtac_base/types/NDPoint.h>
using namespace rtac::types;

int main()
{
    NDPoint<float, 4> p0{0,1,2,3};
    NDPoint<int, 4> p1{0,1,2,3};
    cout << p0 << endl;
    p0 += p1;
    cout << p0 << endl;
    return 0;
}
