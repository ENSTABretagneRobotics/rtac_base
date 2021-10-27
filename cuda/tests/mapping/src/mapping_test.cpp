#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include "mapping_test.h"
using namespace rtac::cuda;

int main()
{
    Texture2D<float> texture(std::move(Texture2D<float>::checkerboard(4,4,0.0f,1.0f)));

    int W = 1024, H = 720;
    
    HostVector<float> rendered0 = render_texture(W,H,texture);
    write_pgm("output0.pgm", W, H, rendered0.data());

    auto map1 = Mapping1::Create(texture);
    HostVector<float> rendered1 = render_mapping1(W,H,map1->device_map());
    write_pgm("output1.pgm", W, H, rendered1.data());

    auto map2 = Mapping2::Create(texture, NormalizerUV({uint2({W,H})}));
    HostVector<float> rendered2 = render_mapping2(W,H,map2->device_map());
    write_pgm("output2.pgm", W, H, rendered2.data());

    return 0;
}
