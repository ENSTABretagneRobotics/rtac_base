#include <iostream>
#include <vector>
#include <math.h>

#include <rtac_tools/types/common.h>
#include <xtensor-blas/xlinalg.hpp>

using namespace std;

template <typename T>
ostream& operator<<(ostream& os, const std::vector<T>& v) {
    if (v.size() < 10) {
        os << "vector :";
        for (auto value : v) {
            os << " " << value;
        }
    }
    else {
        os << "vector (size " << v.size() << ")";
    }
    return os;
}

//template <typename T1, int T2>
//ostream& operator<<(ostream& os, const xt::xtensor<T1,T2>& a) {
//    os << "shape : " << a.shape() << endl;
//    return os;
//}

int main()
{
    vector<float> v{0,1,2,3,4,5,6,7};
    std::vector<size_t> shape{2,4};

    auto a0 = xt::adapt(v, shape);

    cout << v << endl;
    
    for (int i = 0; i < a0.shape()[0]; i++) {
        for (int j = 0; j < a0.shape()[1]; j++) {
            cout << a0(i,j) << " ";
        }
        cout << endl;
    }

    cout << xt::linalg::norm(a0) << endl;

    float theta = 0.25*M_PI;
    xt::xtensor_fixed<float, xt::xshape<2,2>> R = {{cos(theta), -sin(theta)}, {sin(theta), cos(theta)}};
    xt::xtensor_fixed<float, xt::xshape<2,1>> x = {{1,0}};

    auto y = xt::linalg::dot(R,x);
    cout << y << endl;

    return 0;
}
