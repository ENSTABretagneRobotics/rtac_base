#include <iostream>
using namespace std;

#include <rtac_base/containers/ScaledImage.h>
using namespace rtac;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;

template <typename D>
void print(const ScaledImageExpression<D>& img)
{
    cout << "Scaled image :";
    cout << "\n Width dim  :";
    for(auto v : img.width_dim()) { cout << ' ' << v; }
    cout << "\n Height dim :";
    for(auto v : img.height_dim()) { cout << ' ' << v; }
    cout << "\n data :";
    for(int h = 0; h < img.height(); h++) {
        cout << "\n";
        for(int w = 0; w < img.width(); w++) {
            cout << ' ' << img(h,w);
        }
    }
    cout << endl;
}

template <class Derived>
__device__ void process(ScaledImageExpression<Derived>& img)
{
    for(uint32_t h = 0; h < img.height(); h++) {
        for(uint32_t w = 0; w < img.width(); w++) {
            img(h,w) = img.height_dim()[h] * img.width_dim()[w];
        }
    }

}

template <class ImageT>
__global__ void global_process(ImageT img)
{
    static_assert(IsScaledImage<ImageT>::value,
                  "ImageT must be derived from ScaledImageExpression");
    process(img);
}

int main()
{

    uint32_t W = 7, H = 7;
    auto img0 = make_scaled_image(LinearDim(W, {0.0,1.0}),
                                  LinearDim(H, {0.0,1.0}),
                                  HostVector<float>(W*H));
    load_checkerboard(img0, 0.0f, 1.0f);

    auto img1 = make_scaled_image(img0.width_dim(),
                                  img0.height_dim(),
                                  CudaVector<float>(img0.container()));
    global_process<<<1,1>>>(img1.view());
    cudaDeviceSynchronize();
    img0.container() = img1.container();
    print(img0);

    auto img2 = make_scaled_image(
        make_array_dim(CudaVector<float>::linspace(0.0f, 2.0f, W), {0.0f,2.0f}),
        img0.height_dim(), CudaVector<float>(img0.container()));
    global_process<<<1,1>>>(img2.view());
    cudaDeviceSynchronize();
    img0.container() = img2.container();
    print(img0);

    return 0;
}

