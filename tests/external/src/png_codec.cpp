#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_base/external/png_codec.h>
using namespace rtac::external;

int main()
{
    auto path = files::find_one(".*\\.png");
    cout << path << endl;

    PNGCodec codec;
    codec.read_png(path, true);
    codec.read_png(path, false);

    files::write_ppm("output.ppm", codec.width(), codec.height(),
                     (const char*)codec.data().data());

    return 0;
}
