#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/PointFormat.h>
#include <rtac_base/types/Point.h>
#include <rtac_base/containers/HostVector.h>
using namespace rtac;

template <typename T>
void print(const HostVector<T>& v)
{
    auto view = v.view();
    cout << "const_view :";
    for(auto value : view) {
        cout << " " << value;
    }
    cout << endl;
}

template <typename P1, typename P2> inline
HostVector<P1>& copy(HostVector<P1>& dst, const std::vector<P2>& src)
{
    RTAC_ASSERT_COMPATIBLE_POINTS(P1,P2);
    dst.copy_from_host(src.size(), reinterpret_cast<const P1*>(src.data()));
    return dst;
}
struct TestPoint {
    float x, y;
};
namespace rtac {
template<> struct PointFormat<TestPoint> {
    using ScalarType = float;
    static constexpr unsigned int Size = 2;
};
}

int main()
{
    HostVector<int> v0(10);
    for(int i = 0; i < v0.size(); i++) {
        v0[i] = i;
    }
    cout << "v0 : " << v0 << endl;

    HostVector<int> v1(v0);
    cout << "v1 : " << v1 << endl;

    for(auto& value : v1) {
        value *= 2;
    }
    cout << "v1 : " << v1 << endl;

    cout << "size     : " << v0.size()     << endl;
    cout << "capacity : " << v0.capacity() << endl;
    cout << "front    : " << v0.front()    << endl;
    cout << "back     : " << v0.back()     << endl;

    auto view0 = v0.view();
    cout << "view0 :";
    for(auto value : view0) {
        cout << " " << value;
    }
    cout << endl;
    print(v1);

    std::vector<TestPoint> p2(10);
    for(unsigned int i = 0; i < p2.size(); i++) {
        p2[i].x = i; p2[i].y = i + 1;
    }
    HostVector<Point2<float>> p1;
    for(auto p : copy(p1, p2)) {
        std::cout << p << std::endl;
    }

    return 0;
}


