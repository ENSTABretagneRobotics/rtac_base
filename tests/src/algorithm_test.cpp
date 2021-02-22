#include <iostream>
using namespace std;

#include <rtac_base/algorithm.h>
using namespace rtac::algorithm;

using Tuple = std::tuple<float, int, double>;

int main()
{
    cout << "Getting index of some types in std::tuple<float, int, double>" << endl; 
    cout << "Index of float  : " << TupleTypeIndex<float,  Tuple>::value << endl;
    cout << "Index of int    : " << TupleTypeIndex<int,    Tuple>::value << endl;
    cout << "Index of double : " << TupleTypeIndex<double, Tuple>::value << endl;
    cout << "Index of char   : " << TupleTypeIndex<char,   Tuple>::value << endl;

    cout << "Checking if type is in std::tuple<float, int, double>" << endl; 
    cout << "float  : " << TypeInTuple<float,  Tuple>::value << endl;
    cout << "int    : " << TypeInTuple<int,    Tuple>::value << endl;
    cout << "double : " << TypeInTuple<double, Tuple>::value << endl;
    cout << "char   : " << TypeInTuple<char,   Tuple>::value << endl;

    return 0;
}
