#include <iostream>
using namespace std;

#include "mapping_test.h"

int main()
{
    std::vector<float> data(16);
    for(int i = 0; i < data.size(); i++) {
        float x = (2.0*i) / (data.size() - 1) - 1.0f;
        data[i] = 1.0 - x*x;
        cout << " " << data[i];
    }
    cout << endl;
    
    Texture2D<float> texture;
    texture.set_wrap_mode(Texture2D<float>::WrapClamp);
    texture.set_image(data.size(), 1, data.data());

    auto mapping = Mapping<Affine1D,float>::Create(Affine1D({1,0}),
                                                   std::move(texture));
    HostVector<float> x(8);
    for(int i = 0; i < x.size(); i++) {
        x[i] = ((float)i) / (x.size() - 1);
    }
    HostVector<float> out = map(mapping->device_map(), x);
    for(auto v : out) {
        cout << " " << v;
    }
    cout << endl;

    return 0;
}
