#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/Array2D.h>
#include <rtac_base/types/ArrayScale.h>
#include <rtac_base/types/SampledField.h>
using namespace rtac::types;

template <typename T>
struct Affine
{
    T a_;
    T b_;

    static Affine<T> Create(T lb, T ub, unsigned int size) {
        return Affine<T>{(T)((ub - lb) / size), lb};
    }
    
    T operator()(unsigned int idx) const {
        return a_*idx + b_;
    }
};

template <typename T>
std::vector<T> make_data(int N)
{
    std::vector<T> data(N);
    for(int n = 0; n < N; n++) {
        data[n] = n;
    }
    return data;
}
template <typename T>
using ArrayType = Array2D<T,VectorView>;

int main()
{
    int W = 10, H = 6;
    auto data = make_data<float>(W*H);

    SampledField<float, ArrayType, Affine<float>, Affine<float>> field(
        ArrayScale(Affine<float>::Create(0.0f,1.0f,H),
                   Affine<float>::Create(0.0f,1.0f,W)),
                   H,W,data.size(),data.data());

    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            cout << " " << field(h,w);
        }
        cout << endl;
    }

    std::cout << "Coordinates :\n- w :";
    for(int w = 0; w < W; w++) cout << " " << field.get_dimension_coordinate(1, w);
    std::cout << "\n- h : ";
    for(int h = 0; h < H; h++) cout << " " << field.get_dimension_coordinate(0, h);
    cout << endl;
}
