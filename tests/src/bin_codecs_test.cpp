#include <iostream>
#include <random>
using namespace std;

#include <rtac_base/bin_codecs.h>
using namespace rtac;

template <typename T>
inline void print(const std::vector<T>& data)
{
    for(auto v : data) {
        cout << " " << v;
    }
    cout << endl;
}

inline void print(const std::vector<char>& data)
{
    for(auto v : data) {
        cout << v;
    }
    cout << endl;
}

int main()
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0,10.0);

    int N = 10;
    std::vector<double> data(N);
    for(auto& v : data) {
        v = dist(mt);
    }
    print(data);

    std::vector<char> encoded;
    hex_encode(encoded, data);
    print(encoded);

    std::vector<double> decoded;
    hex_decode(decoded, encoded);
    print(decoded);

    return 0;
}


