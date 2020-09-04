#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>

#include <iostream>

void dummy()
{
    using namespace std;
    using namespace rtac::types;

    Pose<float> pose;
    cout << pose << endl;
}
