#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/VectorView.h>
using namespace rtac::types;

template <typename T>
void print_vector(const std::vector<T>& data)
{
    auto view = make_vector_view(data);
    for(auto v : view) {
        cout << " " << v;
    }
    cout << endl;
}

template <typename T>
void fill_vector(std::vector<T>& data)
{
    auto view = make_vector_view(data);
    int count = 0;
    for(auto& v : view) {
        v = count++;
    }
}

int main()
{
    std::vector<float> data(10);
    fill_vector(data);
    print_vector(data);

    return 0;
}

