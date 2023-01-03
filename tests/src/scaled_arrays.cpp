#include <iostream>
using namespace std;

#include <rtac_base/containers/ScaledArray.h>
using namespace rtac;

int main()
{
    for(auto v : LinearDim(11,{-1.0,1.0})) {
        cout << " " << v;
    }
    cout << endl;
    return 0;
}
