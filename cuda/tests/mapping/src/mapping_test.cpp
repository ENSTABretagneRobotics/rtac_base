#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include "mapping_test.h"
using namespace rtac::cuda;

std::vector<float> mapping_data(int N, float k = 2.f)
{
    std::vector<float> output(N);
    //for(int n = 0; n < N; n++) {
    //    float x = 0.9*M_PI*(n - 0.5f*(N-1)) / (N - 1);
    //    output[n] = tan(0.5*x);
    //}

    //float m = output[0], M = output[output.size() - 1];
    //for(int n = 0; n < N; n++) {
    //    m = min(m, output[n]);
    //    M = max(M, output[n]);
    //}

    //for(int n = 0; n < N; n++) {
    //    output[n] = (output[n] - m) / (M - m);
    //}
    for(int n = 0; n < N; n++) {
        float x = 4*(n - 0.5f*(N-1)) / (N - 1);
        output[n] = 1.0 / (1 + exp(-k*x));
    }

    return output;
}

int main()
{

    Texture2D<float> texture(std::move(Texture2D<float>::checkerboard(64,64,0.0f,1.0f)));

    int W = 1024, H = 720;
    
    HostVector<float> rendered0 = render_texture(W,H,texture);
    write_pgm("output0.pgm", W, H, rendered0.data());

    auto map1 = Mapping1::Create(texture);
    HostVector<float> rendered1 = render_mapping1(W,H,map1->device_map());
    write_pgm("output1.pgm", W, H, rendered1.data());

    auto map2 = Mapping2::Create(texture, NormalizerUV({uint2({W,H})}));
    HostVector<float> rendered2 = render_mapping2(W,H,map2->device_map());
    write_pgm("output2.pgm", W, H, rendered2.data());


    auto v = mapping_data(256);
    auto map3 = Mapping3::Create(0,1,v.data(), v.size());
    HostVector<float> rendered3 = render_mapping3(W,H,map1->device_map(), map3->device_map());
    write_pgm("output3.pgm", W, H, rendered3.data());

    return 0;
}
