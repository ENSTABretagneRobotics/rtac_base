#include <iostream>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/external/kalibr.h>
using namespace rtac;
using namespace rtac::external;

int main()
{
    auto filename = files::find_one(".*camchain-calibration_selection.yaml");
    std::cout << "Filename : " << filename << std::endl;

    KalibrCalibration calibration(filename);
    cout << calibration << endl;

    return 0;
}
