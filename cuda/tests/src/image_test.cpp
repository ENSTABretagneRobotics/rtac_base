#include <iostream>
using namespace std;

#include <rtac_base/types/Image.h>
using namespace rtac::types;

#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;

int main()
{
    Image<unsigned int> img({32,32});
    cout << img << endl;
    cout << img.data().size() << endl;


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
    for(int h = 0; h < himg.height(); h++) {
        for(int w = 0; w < himg.width(); w++) {
            cout << himg(h,w) << ' ';
        }
        cout << endl;
    }


    return 0;
}
