#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/cuda/reductions.hcu>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

DeviceVector<unsigned int> image_data(unsigned int W, unsigned int H)
{
    std::vector<unsigned int> data(W*H);
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            data[W*h + w] = h + 1;
        }
    }
    return data;
}

int main()
{
    unsigned int N = 123456789;

    DeviceVector<float> inF(std::vector<float>(N, 1));
    device::reduce(inF.data(), inF.data(), N);
    cout << inF << endl;

    DeviceVector<unsigned int> inU(std::vector<unsigned int>(N, 1));
    device::reduce(inU.data(), inU.data(), N);
    cout << inU << endl;
    
    unsigned int W = 111111, H = 9;
    auto img = image_data(W,H);
    DeviceVector<unsigned int> output(H);
    device::reduce_lines(img.data(), output.data(), W, H);
    cout << output << endl;

    return 0;
}
