#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/SharedVector.h>
using namespace rtac::types;
using Vector = SharedVector<float>;

constexpr const auto t0 = ensure_shared_vector<Vector>();
// next line will successfully fail at compile time
// constexpr const auto t1 = ensure_shared_vector<std::vector<float>>();

std::ostream& operator<<(std::ostream& os, const Vector& v)
{
    auto data = v.data();
    os << "(";
    if(v.size() <= 10) {
        os << data[0];
        for(int i = 1; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    else {
        for(int i = 1; i < 3; i++) {
            os << data[i] << " ";
        }
        os << "...";
        for(int i = v.size() - 3; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    os << ")";
    return os;
}

int main()
{
    Vector v0(10);
    for(auto& value : v0) {
        value = 3;
    }
    cout << "v0 : " << v0 << endl;

    Vector v1(v0);
    cout << "Shallow copy init :" << endl;
    cout << "v0 : " << v0 << endl;
    cout << "v1 : " << v1 << endl;
    v0.data()[0] = 1;
    cout << "Shallow copy v0 modified :" << endl;
    cout << "v0 : " << v0 << endl;
    cout << "v1 : " << v1 << endl;

    Vector v2(v0.copy());
    cout << "Deep copy init :" << endl;
    cout << "v0 : " << v0 << endl;
    cout << "v2 : " << v2 << endl;
    v0.data()[1] = 1;
    cout << "Deep copy v0 modified :" << endl;
    cout << "v0 : " << v0 << endl;
    cout << "v2 : " << v2 << endl;

    return 0;
}
