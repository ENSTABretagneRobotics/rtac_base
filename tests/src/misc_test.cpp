#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Rectangle.h>
#include <rtac_base/time.h>
using FrameCounter = rtac::time::FrameCounter;

int main()
{
    rtac::Shape<size_t> shape({800,600});
    cout << "Shape : " << shape << endl;
    cout << "Shape.ratio<float>() : " << shape.ratio<float>() << endl;
    cout << "Shape.area()         : " << shape.area() << endl;

    rtac::Rectangle<size_t> rect({0,2,3,6});
    cout << "Rectangle : " << rect << endl;
    cout << "Rectangle.shape() " << rect.shape() << endl;

    FrameCounter counter;
    for(int i = 0; i < 100; i++) {
        cout << counter << flush;
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}
