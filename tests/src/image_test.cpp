#include <iostream>
using namespace std;

#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Image.h>
using namespace rtac::types;

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
    for(int i = 0; i < img.size(); i++) {
        cout << img[i] << ' ';
    }
    cout << endl;

    return 0;
}
