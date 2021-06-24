#include <iostream>
using namespace std;

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

#include <rtac_base/types/common.h>
#include <rtac_base/interpolation.h>
using namespace rtac::algorithm;
using namespace rtac::types::indexing;



Interpolator<float>::Vector linspace(float vmin, float vmax, unsigned int N)
{
    Interpolator<float>::Vector output(N);
    for(int n = 0; n < N; n++) {
        output[n] = (vmax - vmin) * n / (N - 1) + vmin;
    }
    return output;
}

Interpolator<float>::Matrix dickins_data()
{
    Interpolator<float>::Matrix data(23,2);
    data <<   0, 1476.7,
             38, 1476.7,
             50, 1472.6,
             70, 1468.8,
            100, 1467.2,
            140, 1471.6,
            160, 1473.6,
            170, 1473.6,
            200, 1472.7,
            215, 1472.2,
            250, 1471.6,
            300, 1471.6,
            370, 1472.0,
            450, 1472.7,
            500, 1473.1,
            700, 1474.9,
            900, 1477.0,
           1000, 1478.1,
           1250, 1480.7,
           1500, 1483.8,
           2000, 1490.5,
           2500, 1498.3,
           3000, 1506.5;
    return data;
}

std::vector<float> to_vector(const Interpolator<float>::Vector& v)
{
    std::vector<float> out(v.rows());
    for(int i = 0; i < out.size(); i++) {
        out[i] = v(i);
    }
    return out;
}

std::vector<float> to_vector(const Interpolator<float>::Matrix& v)
{
    std::vector<float> out(v.rows());
    for(int i = 0; i < out.size(); i++) {
        out[i] = v(i,0);
    }
    return out;
}

int main()
{
    auto data = dickins_data();
    Interpolator<float>::Vector x0 = data(all,0);
    Interpolator<float>::Vector y0 = data(all,1);

    auto x = linspace(data(0,0), data(last,0), 512);

    InterpolatorNearest<float> interpNN(data(all,0), data(all,1));
    auto ynn = interpNN(x);
    InterpolatorLinear<float> interpLinear(data(all,0), data(all,1));
    auto yl = interpLinear(x);
    
    plt::plot(to_vector(x0), to_vector(y0),  {{"label", "original"}});
    plt::plot(to_vector(x),  to_vector(ynn), {{"label", "nearest"}});
    plt::plot(to_vector(x),  to_vector(yl),  {{"label", "linear"}});
    plt::legend();
    plt::show();

    return 0;
}
