#include <iostream>
#include <vector>
using namespace std;

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

    return 0;
}


