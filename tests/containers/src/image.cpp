#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/Shape.h>
#include <rtac_base/containers/Image.h>
#include <rtac_base/containers/utilities.h> // optional
using namespace rtac;

template <typename T, template<typename> class C>
void print_image(const Image<T,C>& img)
{
    //auto view = img.view();
    auto view = make_view(img);
    cout << view << endl;
    cout << view.size() << endl;
    for(int i = 0; i < view.size(); i++) {
        cout << view[i] << ' ';
    }
    cout << endl;
}

int main()
{
    Image<unsigned int, std::vector> img(32,32);
    for(int h = 0; h < img.height(); h++) {
        for(int w = 0; w < img.width(); w++) {
            img(h,w) = img.width()*h + w;
        }
    }

    cout << img << endl;
    cout << img.size() << endl;
    for(int i = 0; i < img.size(); i++) {
        cout << img[i] << ' ';
    }
    cout << endl;

    Image<unsigned int> img1(img);
    cout << img1 << endl;
    cout << img1.size() << endl;
    for(int i = 0; i < img1.size(); i++) {
        cout << img1[i] << ' ';
    }
    cout << endl;

    print_image(img1);

    return 0;
}
