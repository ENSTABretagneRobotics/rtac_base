#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/Texture2D.h>
#include <rtac_base/cuda/TextureView2D.h>
using namespace rtac::cuda;


__global__ void render_texture(float* dst, TextureView1D<float> src)
{
    printf("%f ", ((float)threadIdx.x) / blockDim.x);
    dst[threadIdx.x] = src(((float)threadIdx.x) / blockDim.x);
}

//__global__ void render_texture(float* dst, TextureView2D<float> src)
//{
//    dst[threadIdx.x] = src(((float)threadIdx.x) / blockDim.x, 0);
//}

int main()
{
    std::vector<float> data(16);
    for(int i = 0; i < data.size(); i++) { data[i] = i; }
    
    Texture2D<float> texData;
    texData.set_image(data.size(), 1, data.data());

    DeviceVector<float> output(16);
    render_texture<<<1,16>>>(output.data(), texData.view1d());
    cudaDeviceSynchronize();
    cout << endl;

    cout << output << std::endl;

    return 0;
}
