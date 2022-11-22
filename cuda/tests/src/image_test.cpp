#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/Image.h>
using namespace rtac;

#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;

template <typename T, template<typename>class C>
void print_image(const Image<T,C>& img)
{
    auto view = img.view();
    for(int h = 0; h < view.height(); h++) {
        for(int w = 0; w < view.width(); w++) {
            cout << view(h,w) << ' ';
        }
        cout << endl;
    }
}

int main()
{
    Image<unsigned int, std::vector> img({32,32});
    cout << img << endl;
    cout << img.size() << endl;


    for(int h = 0; h < img.height(); h++) {
        for(int w = 0; w < img.width(); w++) {
            img(h,w) = img.width()*h + w;
        }
    }

    Image<unsigned int, DeviceVector> dimg;
    dimg = img;

    // Does not work (as expected)
    //cout << dimg(0,0) << endl;
    
    Image<unsigned int, HostVector> himg(dimg);
    print_image(himg);

    return 0;
}
