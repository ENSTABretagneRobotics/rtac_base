#include <iostream>

#include <rtac_tools/files.h>

using namespace std;
namespace nf = rtac::files;

int main()
{
    cout << "RTAC_DATA : " << nf::rtac_data_path() << endl;
    for(auto& path : nf::find()) {
        cout << path << endl;
    }
    cout << endl;
    for(auto& path : nf::find(".*\\.tif")) {
        cout << path << endl;
    }
    cout << endl;

    cout << nf::find_one(".*ortho.*\\.tif") << endl << endl;
    return 0;
}


