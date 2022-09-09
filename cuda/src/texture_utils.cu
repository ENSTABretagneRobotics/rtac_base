#include <rtac_base/cuda/texture_utils.h>

#include <rtac_base/cuda_defines.h>

namespace rtac { namespace cuda {

using namespace rtac::types;

template <typename T>
__global__ void render_texture_kernel(TextureView2D<T> tex, ImageView<T> out)
{
    uint32_t w = blockDim.x*blockIdx.x + threadIdx.x;
    for(; w < tex.width_; w += gridDim.x*blockDim.x) {
        out(blockIdx.y, w) = tex(w, blockIdx.y);
    }
}


template <typename T>
void do_render_texture(const Texture2D<T>& tex, rtac::types::ImageView<T> out)
{
    render_texture_kernel<<<{out.width() / RTAC_BLOCKSIZE + 1, out.height()}, RTAC_BLOCKSIZE>>>(
        tex.view(), out);
    cudaDeviceSynchronize();
}

void render_texture(const Texture2D<float>& tex, rtac::types::ImageView<float> out)
{
    do_render_texture(tex, out);
}

void render_texture(const Texture2D<float2>& tex, rtac::types::ImageView<float2> out)
{
    do_render_texture(tex, out);
}

void render_texture(const Texture2D<float4>& tex, rtac::types::ImageView<float4>& out)
{
    do_render_texture(tex, out);
}


} //namespace cuda
} //namespace rtac

