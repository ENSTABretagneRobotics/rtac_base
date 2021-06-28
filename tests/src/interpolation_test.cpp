#include <iostream>
using namespace std;

#include <matplotlibcpp.h>
namespace plt = matplotlibcpp;

#include <rtac_base/types/common.h>
#include <rtac_base/interpolation.h>
template <typename T>
using Vector = rtac::types::Vector<T>;
using namespace rtac::algorithm;
using namespace rtac::types::indexing;



template <typename VectorT>
VectorT linspace(float vmin, float vmax, unsigned int N)
{
    VectorT output(N);
    for(int n = 0; n < N; n++) {
        output[n] = (vmax - vmin) * n / (N - 1) + vmin;
    }
    return output;
}

Eigen::MatrixXf dickins_data()
{
    Eigen::MatrixXf data(23,2);
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

const std::vector<float>& to_vector(const std::vector<float>& v)
{
    return v;
}

std::vector<float> to_vector(const rtac::types::Vector<float>& v)
{
    std::vector<float> out(v.size());
    for(int i = 0; i < out.size(); i++) {
        out[i] = v(i);
    }
    return out;
}

int main()
{
    auto data = dickins_data();
    auto x0 = data(all,0);
    auto y0 = data(all,1);

    auto x = linspace<Vector<float>>(data(0,0), data(last,0), 8192);

    InterpolatorNearest<float> interpNN(x0, y0);
    auto ynn = interpNN(x);
    InterpolatorLinear<float> interpLinear(x0, y0);
    auto yl = interpLinear(x);
    InterpolatorCubicSpline<float> interpCubicSpline(x0, y0);
    auto yc = interpCubicSpline(x);
    
    plt::plot(to_vector(x0), to_vector(y0),  {{"marker", "o"},
                                              {"label", "original"}});
    plt::plot(to_vector(x),  to_vector(ynn), {{"label", "nearest"}});
    plt::plot(to_vector(x),  to_vector(yl),  {{"label", "linear"}});
    plt::plot(to_vector(x),  to_vector(yc),  {{"label", "cubic"}});
    plt::legend();
    plt::show();

    return 0;
}
