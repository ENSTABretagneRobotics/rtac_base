#include <iostream>
using namespace std;

#include <rtac_base/types/Pose.h>
using namespace rtac;

int main()
{
    {
        std::string poseStr = "quat, 1.0, 2.0, 3.0, 0.0, 1.0, 0.0, 0.0";
        auto p = Pose<float>::decode_string(poseStr);
        std::cout << p << std::endl << p.encode_string("quat", ',') << std::endl;
    }
    {
        std::string poseStr =
            "hmat, 0.0, 1.0, 0.0, 1.0,-1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0";
        auto p = Pose<float>::decode_string(poseStr);
        std::cout << p << std::endl << p.encode_string("hmat", ',') << std::endl;
    }
    return 0;
}
