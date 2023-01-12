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

    int N = 4;
    std::vector<double> data(N);
    //for(auto& v : data) {
    //    v = dist(mt);
    //}
    for(int n = 0; n < N; n++) {
        data[n] = n + 1000;
    }
    print(data);

    std::vector<char> encoded;
    hex_encode(encoded, data);
    cout << "encoded size : " << encoded.size() << endl;
    print(encoded);
    cout << "0000000000408F400000000000488F400000000000508F400000000000588F40" << endl;

    std::vector<double> decoded;
    hex_decode(decoded, encoded);
    print(decoded);

    return 0;
}


