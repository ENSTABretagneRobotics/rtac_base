#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/DimExpression.h>
#include <rtac_base/containers/Image.h>
#include <rtac_base/containers/ScaledImage.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
#include <rtac_base/cuda/TextureFunction2D.h>
using namespace rtac::cuda;

template <class D1, class D2>
__device__ void do_render_function(ScaledImageExpression<D1>& dst, const Function2D<D2>& src)
{
    for(int h = 0; h < dst.height(); h++) {
        for(int w = 0; w < dst.width(); w++) {
            printf("%f %f ", dst.width_dim()[w], dst.height_dim()[h]);
            dst(h,w) = src(dst.width_dim()[w], dst.height_dim()[h]);
        }
    }
}

// pass by value to avoid slicing
template <class ScaledImageT, class Function2DT>
__global__ void render_function(ScaledImageT dst, Function2DT src)
{
    do_render_function(dst, src);
}

template <typename D> inline
void print(const ImageExpression<D>& img)
{
    for(int h = 0; h < img.height(); h++) {
        for(int w = 0; w < img.width(); w++) {
            cout << ' ' << img(h,w);
        }
        cout << endl;
    }
}

int main()
{
    unsigned int W = 32, H = 32;
    auto dst = make_scaled_image(LinearDim(2*W, {-1.0f,1.0f}),
                                 LinearDim(2*H, {-1.0f,1.0f}),
                                 CudaVector<float>(4*W*H));
    auto srcData = Texture2D<float>::checkerboard(W,H,0.0f,1.0f);
    srcData.set_filter_mode(Texture2D<float>::FilterLinear);

    render_function<<<1,1>>>(dst.view(),
        make_texture_function(srcData, Bounds<float>{.0f, 1.0f},
                                       Bounds<float>{.0f, 1.0f}));
    cudaDeviceSynchronize();

    print(Image<float,HostVector>(dst.shape(), dst.container()));

    return 0;
}


