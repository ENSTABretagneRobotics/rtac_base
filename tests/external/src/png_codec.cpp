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
    codec.read_image(path);
    files::write_ppm("output_png.ppm", codec.width(), codec.height(),
                     (const char*)codec.data().data());

    std::cout << "Bit depth : " << codec.bitdepth() << std::endl;

    codec.read_image(path, true);
    files::write_ppm("output_png_inverted.ppm", codec.width(), codec.height(),
                     (const char*)codec.data().data());

    return 0;
}
