#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/VectorView.h>
using namespace rtac;

template <typename T>
void print_vector(const std::vector<T>& data)
{
    //auto view = VectorView(data.size(), data.data());
    auto view = make_view(data);
    for(auto v : view) {
        cout << " " << v;
    }
    cout << endl;
    for(int i = 0; i < view.size(); i++) {
        cout << " " << view[i];
    }
    cout << endl;

    cout << "Front : " << view.front() << endl;
    cout << "Back  : " << view.back()  << endl;
}

template <typename T>
void fill_vector(std::vector<T>& data)
{
    //auto view = VectorView(data.size(), data.data());
    auto view = make_view(data);
    int count = 0;
    for(auto& v : view) {
        v = count++;
    }
    for(int i = 0; i < view.size(); i++) {
        view[i] = 2*view[i];
    }

    view.front() = 100;
    view.back()  = 101;
}

int main()
{
    std::vector<float> data(10);
    //for(int i = 0; i < data.size(); i++) data[i] = i;
    fill_vector(data);
    print_vector(data);

    return 0;
}

