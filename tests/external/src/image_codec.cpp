#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac;

#include <rtac_base/external/ImageCodec.h>
using namespace rtac::external;

int main()
{
    auto path0 = files::find_one(".*\\.png");
    auto path1 = files::find_one(".*\\.jpg");
    cout << path0 << endl;
    cout << path1 << endl;

    ImageCodec codec;
    auto img0 = codec.read_image(path0);
    img0->write_image("output0.png", {img0->width(), img0->height(), img0->step(), img0->bitdepth(), img0->channels()}, img0->data().data(), true);
    //files::write_ppm("output0.ppm", img0->width(), img0->height(),
    //                 (const char*)img0->data().data());

    auto img1 = codec.read_image(path1);
    files::write_ppm("output1.ppm", img1->width(), img1->height(),
                     (const char*)img1->data().data());

    return 0;
}
