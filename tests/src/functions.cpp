#include <iostream>
using namespace std;

#include <rtac_base/types/functions.h>
using namespace rtac;

int main()
{
    float a = 2.0;
    float b = 0.0;

    auto func0 = LinearFunction1D::make(a, b, Bounds<float>{0.0f,1.0f});

    int N = 10;
    for(int n = 0; n < N + 1; n++) {
        float x = ((float)n) / N;
        cout << ' ' << func0(x);
    }
    cout << endl;

    return 0;
}


