#include <iostream>
#include <complex>
using namespace std;

#include <rtac_base/types/Complex.h>
using namespace rtac;

int main()
{
    Complex<float> a(1,2);
    Complex<float> b(3,4);

    cout <<  a << " " <<  b << endl;
    cout << +a << " " << -b << endl;

    cout << "operator+ : " << a + b << endl;
    cout << "operator- : " << a - b << endl;
    cout << "operator* : " << a * b << endl;
    cout << "operator/ : " << a / b << endl;

    cout << "operator+ : " << a + 2.0f << endl;
    cout << "operator- : " << a - 2.0f << endl;
    cout << "operator* : " << a * 2.0f << endl;
    cout << "operator/ : " << a / 2.0f << endl;

    cout << "operator+ : " << 2.0f + a << endl;
    cout << "operator- : " << 2.0f - a << endl;
    cout << "operator* : " << 2.0f * a << endl;
    cout << "operator/ : " << 2.0f / a << endl;

    cout << "abs  : " <<  abs(a) << " " <<  abs(b) << endl;
    cout << "norm : " << norm(a) << " " << norm(b) << endl;
    cout << "arg  : " <<  arg(a) << " " <<  arg(b) << endl;

    cout << "abs(a*b)        : " << abs(a*b)        << endl;
    cout << "abs(a) * abs(b) : " << abs(a) * abs(b) << endl;
    cout << "arg(a*b)        : " << arg(a*b)        << endl;
    cout << "arg(a) + arg(b) : " << arg(a) + arg(b) << endl;

    cout << "abs(a/b)        : " << abs(a/b)        << endl;
    cout << "abs(a) / abs(b) : " << abs(a) / abs(b) << endl;
    cout << "arg(a/b)        : " << arg(a/b)        << endl;
    cout << "arg(a) - arg(b) : " << arg(a) - arg(b) << endl;

    return 0;
}

