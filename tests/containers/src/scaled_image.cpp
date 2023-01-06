#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/ScaledImage.h>
using namespace rtac;

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

int main()
{
    for(auto v : LinearDim(11,{-1.0,1.0})) {
        cout << " " << v;
    }
    cout << endl;

    struct Traits {
        using value_type = float;
        using Container  = HostVector<float>;
        using WidthDim   = LinearDim;
        using HeightDim  = LinearDim;
    };
    uint32_t W = 32, H = 32;
    auto img0 = ScaledImage<float, LinearDim, LinearDim, HostVector>(LinearDim(W, {-1.0,1.0}),
                                                                     LinearDim(H, {-1.0,1.0}),
                                                                     HostVector<float>(W*H));
    auto img1 = make_scaled_image(LinearDim(W, {-1.0,1.0}),
                                  LinearDim(H, {0,H-1.0f}),
                                  HostVector<float>(W*H));
    load_checkerboard(img1, 0.0f, 1.0f);

    cout << "img1.width_dim : ";
    for(auto v : img1.width_dim()) {
        cout << ' ' << v;
    }
    cout << endl;
    cout << "img1.height_dim : ";
    for(auto v : img1.height_dim()) {
        cout << ' ' << v;
    }
    cout << endl;

    print(img1);

    return 0;
}


