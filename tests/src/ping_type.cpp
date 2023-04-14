#include <iostream>

#include <rtac_base/types/SonarPing.h>
using namespace rtac;

template <typename T>
void const_print(const Ping2D<T>& p) {
    std::cout << "const ping view  : " << p.view() << std::endl;
}

template <typename T>
void print(Ping2D<T>& p) {
    std::cout << "ping view  : " << p.view() << std::endl;
}

int main()
{
    Ping2D<float> p0(Linspace<float>(0.0f, 10.0f, 512),
                     HostVector<float>::linspace(-0.25*3.14, 0.25*3.14, 512));
    std::cout << "p0 : " << p0 << std::endl;

    const_print(p0);
    print(p0);

    return 0;
}

