#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/types/Time.h>
using Clock = rtac::types::SteadyClock;

int main()
{
    for(int i = 0; i < 1000; i++) {
        cout << Clock::now().milliseconds() << endl;
        this_thread::sleep_for(100ms);
    }
    return 0;
}
