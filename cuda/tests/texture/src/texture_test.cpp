#include <iostream>
#include <vector>
#include <functional>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/Texture2D.h>
using namespace rtac;
using namespace rtac::cuda;

#include "texture_test.h"


int main()
{
    int W    = 16,  H    = 16;  // Texture size.
    int Wout = 512, Hout = 512; // Rendering size to test for filtering mode
                                // (interpolation on texture fetch).
    int W2   = W/2, H2   = H/2; // Subportion of texture to write to.

    // Monocomponent texture
    auto tex0 = Texture2D<float>::checkerboard(W, H, 0.0f, 1.0f);

    HostVector<float> res0;
    res0 = render_texture(Wout,Hout,tex0);

    std::vector<float> subdata0(W2*H2, 0.5f);
    tex0.set_subimage(W2,H2,W2,H2, subdata0.data());
    res0 = render_texture(Wout,Hout,tex0);

    //write_pgm("output.pgm", Wout, Hout, res0.data());

    // Colored texture
    auto tex1 = Texture2D<float4>::checkerboard(W, H,
                                                float4({0.0f,0.0f,1.0f,1.0f}),
                                                float4({0.0f,1.0f,0.0f,1.0f}));

    HostVector<float4> res1;
    res1 = render_texture(Wout,Hout,tex1);

    std::vector<float4> subdata1(W2*H2, float4({0.5f,0.5f,0.5f,1.0f}));
    tex1.set_subimage(W2,H2,W2,H2, subdata1.data());
    res1 = render_texture(Wout,Hout,tex1);
    // testing texture copy
    Texture2D<float4> tex2(tex1);
    tex2.set_subimage(W2,H2,0,0, subdata1.data());

    auto rgbData = std::vector<uint8_t>(3*res1.size());
    for(int i = 0, j = 0; i < res1.size(); i++) {
        rgbData[j]     = 255*res1[i].x;
        rgbData[j + 1] = 255*res1[i].y;
        rgbData[j + 2] = 255*res1[i].z;
        j += 3;
    }
    write_ppm("output.ppm", Wout, Hout, (const char*)rgbData.data());

    HostVector<float4> res2;
    res2 = render_texture(Wout,Hout,tex2);
    rgbData.resize(3*res2.size());
    for(int i = 0, j = 0; i < res2.size(); i++) {
        rgbData[j]     = 255*res2[i].x;
        rgbData[j + 1] = 255*res2[i].y;
        rgbData[j + 2] = 255*res2[i].z;
        j += 3;
    }
    write_ppm("output_copy.ppm", Wout, Hout, (const char*)rgbData.data());


    Texture2D<float4> tex3(std::move(tex2));
    res2 = render_texture(Wout,Hout,tex3);
    rgbData.resize(3*res2.size());
    for(int i = 0, j = 0; i < res2.size(); i++) {
        rgbData[j]     = 255*res2[i].x;
        rgbData[j + 1] = 255*res2[i].y;
        rgbData[j + 2] = 255*res2[i].z;
        j += 3;
    }
    write_ppm("output_move.ppm", Wout, Hout, (const char*)rgbData.data());

    return 0; 
}


