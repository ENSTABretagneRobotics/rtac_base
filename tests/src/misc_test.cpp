#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/misc.h>
using FrameCounter = rtac::misc::FrameCounter;

int main()
{
    FrameCounter counter;
    for(int i = 0; i < 100; i++) {
        cout << counter << flush;
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}
