#include "texture_test.h"

namespace rtac { namespace cuda {

template <typename T>
__global__ void do_render(T* output, size_t width, cudaTextureObject_t texObject)
{
    for(int i = threadIdx.x; i < width; i += blockDim.x) {
        float x = ((float)i) / width;
        float y = ((float)blockIdx.x) / gridDim.x;
        output[width*blockIdx.x + i] = tex2D<T>(texObject, x, y);
    }
}

template <typename T>
CudaVector<T> render(size_t width, size_t height,
                       const Texture2D<T>& texObject)
{
    CudaVector<T> res(width*height);

    do_render<T><<<height, 512>>>(res.data(), width, texObject.texture());
    cudaDeviceSynchronize();

    return res;
}

CudaVector<float> render_texture(size_t width, size_t height,
                                 const Texture2D<float>& texObject)
{
    return render<float>(width, height, texObject);
}

CudaVector<float4> render_texture(size_t width, size_t height,
                                  const Texture2D<float4>& texObject)
{
    return render<float4>(width, height, texObject);
}

}; //namespace cuda
}; //namespace rtac
