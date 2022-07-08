#include <iostream>
using namespace std;

#include <rtac_base/types/ArrayScale.h>
using namespace rtac::types;

template <typename T>
class Affine : public ScaleFunctor<Affine<T>>
{
    public:

    using value_type = T;

    protected:

    T a_;
    T b_;

    public:

    Affine(T a, T b) : a_(a), b_(b) {}
    static Affine<T> Create(T lowerBound, T upperBound, std::size_t size) {
        auto extent = upperBound - lowerBound;
        return Affine<T>(extent / size, lowerBound + extent / (2*size));
    }

    RTAC_HOSTDEVICE T operator()(std::size_t idx) const { return a_*idx + b_; }
};

int main()
{
    int N = 10;

    auto affine = Affine<float>::Create(0.0f, 1.0f, 10);
    for(int n = 0; n < N; n++) {
        cout << " " << affine(n);
    }
    cout << endl;

    ArrayScale scales(Affine<float>::Create(0.0f, 1.0f, 10),
                      Affine<float>::Create(0.0f, 2.0f, 10));
    for(int n = 0; n < N; n++) {
        cout << scales.scale<0>()(n) << " "
             << scales.scale<1>()(n) << endl;
    }
    
    cout << "Using ArrayScale :\n";
    for(int n = 0; n < N; n++) {
        auto res = scales(n,n);
        cout << res[0] << " " << res[1] << endl;
    }

    return 0;
}
