#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_base/external/jpg_codec.h>
using namespace rtac::external;

int main()
{
    auto path = files::find_one(".*\\.jpg");
    cout << path << endl;

    JPGCodec codec;
    codec.read_jpg(path);
    codec.read_jpg(path, true);

    files::write_ppm("output_inverted.ppm", codec.width(), codec.height(),
                     (const char*)codec.data().data());

    return 0;
}
