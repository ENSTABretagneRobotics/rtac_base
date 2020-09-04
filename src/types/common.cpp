#include <rtac_tools/types/common.h>
#include <rtac_tools/types/Pose.h>

#include <iostream>

void dummy()
{
    using namespace std;
    using namespace rtac::types;

    Pose<float> pose;
    cout << pose << endl;
}
