#include <iostream>
#include <sstream>
using namespace std;

#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Shape.h>
using namespace rtac;

#include <rtac_base/ply_files.h>
using namespace rtac;

int main()
{
    auto data = ply::New();
    cout << data << endl;

    ply::add_pose(data, Pose<float>::Identity());
    ply::add_shape(data, Shape<uint32_t>({256,480}));
    ply::add_rectangle(data, Rectangle<double>({1,2,3,4}));
    std::ostringstream oss;
    oss << data;

    std::istringstream iss(oss.str());
    auto reloaded = ply::read(iss);
    cout << reloaded << endl;
    cout << ply::get_pose<float>(reloaded) << endl;
    cout << ply::get_shape<uint32_t>(reloaded) << endl;
    cout << ply::get_rectangle<double>(reloaded) << endl;

    return 0;
}




