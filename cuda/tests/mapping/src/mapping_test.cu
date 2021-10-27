#include "mapping_test.h"
#include "mapping_test.hcu"

DeviceVector<float> render_texture(int W, int H, const Texture2D<float>& texData)
{
    DeviceVector<float> output(W*H);
    
    do_render_texture<<<1,1>>>(output.data(), W, H, texData.texture());
    cudaDeviceSynchronize();

    return output;
}

DeviceVector<float> render_mapping1(int W, int H, const Mapping1::DeviceMap& map)
{
    DeviceVector<float> output(W*H);
    
    do_render_mapping1<<<1,1>>>(output.data(), W, H, map);
    cudaDeviceSynchronize();

    return output;
}

DeviceVector<float> render_mapping2(int W, int H, const Mapping2::DeviceMap& map)
{
    DeviceVector<float> output(W*H);
    
    do_render_mapping2<<<1,1>>>(output.data(), W, H, map);
    cudaDeviceSynchronize();

    return output;
}

DeviceVector<float> render_mapping3(int W, int H,
                                    const Mapping1::DeviceMap& map1,
                                    const Mapping3::DeviceMap& map3)
{
    DeviceVector<float> output(W*H);
    
    do_render_mapping3<<<1,1>>>(output.data(), W, H, map1, map3);
    cudaDeviceSynchronize();

    return output;
}

