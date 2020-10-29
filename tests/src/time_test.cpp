#include <iostream>
#include <thread>
using namespace std;

#include <rtac_base/types/Time.h>
using Clock = rtac::types::Clock<>;

int main()
{
    auto t1 = Clock::now();
    auto t2 = t1;
    for(int i = 0; i < 1000; i++) {
        t2 = Clock::now();
        cout << t2 << " " << (t2 > t1) << (t2 < t1) << (t2 == t2) << endl;
        this_thread::sleep_for(100ms);
    }
    return 0;
}
