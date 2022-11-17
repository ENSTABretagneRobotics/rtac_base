#include <iostream>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/external/obj_codec.h>
using namespace rtac;

int main()
{
    auto datasetPath = "/home/pnarvor/work/narval/data/others/fabien/";
    cout << "Dataset path : " << datasetPath << endl;

    external::ObjLoader parser(datasetPath);
    parser.load_geometry();

    cout << parser << endl;
    cout << parser.bounding_box() << endl;

    getchar();

    return 0;
}

