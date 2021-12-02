#include <iostream>
using namespace std;

#include <rtac_base/cuda/Matrix.h>
using namespace rtac::cuda::linear;

int main()
{
    Matrix3<float> R;
    R << 1,2,3,
         4,5,6,
         7,8,9;
    cout << R << endl;
    return 0;
}
