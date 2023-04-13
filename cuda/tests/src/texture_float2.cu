#include <iostream>
using namespace std;

#include <rtac_base/files.h>
#include <rtac_base/containers/Image.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
#include <rtac_base/cuda/Texture2D.h>
#include <rtac_base/cuda/texture_utils.h>
using namespace rtac::cuda;

struct Color
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

void write_image(const Image<float2>& img)
{
    Image<Color> tmp(img.shape());
    for(unsigned int h = 0; h < img.height(); h++) {
        for(unsigned int w = 0; w < img.width(); w++) {
            tmp(h,w).r = 255*img(h,w).x;
            tmp(h,w).g = 255*img(h,w).y;
            tmp(h,w).b = 0;
        }
    }
    
    files::write_ppm("output.ppm", tmp.width(), tmp.height(), (const char*)tmp.data());
}

int main()
{
    auto tex = Texture2D<float2>::checkerboard(512,512,{1,0},{0,1});
    Image<float2,CudaVector> img(tex.width(), tex.height());
    render_texture(tex, img.view());

    //Image<float2,HostVector> himg(img);
    write_image(img);

    return 0;
}
