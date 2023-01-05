#include <iostream>
using namespace std;

#include <rtac_base/containers/utilities.h>
using namespace rtac;

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const VectorView<T>& view) {
    os << view[0];
    for(auto v : view) {
        os << ' ' << v;
    }
    return os;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vect) {
    os << vect[0];
    for(auto v : vect) {
        os << ' ' << v;
    }
    return os;
}

int main()
{
    auto v0 = HostVector<float>::linspace(0,10,11);
    cout << "v0    : " << v0 << endl;
    auto view0 = make_view(v0);
    cout << "view0 : " << view0 << endl;

    std::vector<float> v1(v0.size());
    v0.copy_to(v1.data());
    cout << "v1    : " << v1 << endl;
    auto view1 = make_view(v1);
    cout << "view1 : " << view1 << endl;


    return 0;
}
