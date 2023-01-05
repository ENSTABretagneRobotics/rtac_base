#include <iostream>
using namespace std;

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/containers/ScaledArray.h>
using namespace rtac;

int main()
{
    for(auto v : LinearDim(11,{-1.0,1.0})) {
        cout << " " << v;
    }
    cout << endl;

    struct Config {
        using value_type = float;
        using Container  = HostVector<float>;
        using WidthDim   = LinearDim;
        using HeightDim  = LinearDim;
    };
    uint32_t W = 256, H = 256;
    auto img0 = ScaledImage<Config>(HostVector<float>(W*H),
                                    LinearDim(256, {-1.0,1.0}),
                                    LinearDim(256, {-1.0,1.0}));
    auto img1 = make_scaled_image(HostVector<float>(W*H),
                                  LinearDim(256, {-1.0,1.0}),
                                  LinearDim(256, {-1.0,1.0}));
    return 0;
}
