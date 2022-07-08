#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/Array2D.h>
using namespace rtac::types;

template <typename T>
void print_view(Array2D<T,VectorView> view)
{
    for(int i = 0; i < view.rows(); i++) {
        for(int j = 0; j < view.cols(); j++) {
            cout << " " << view(i,j);
        }
        cout << endl;
    }
}

template <typename ArrayT>
void print_array(const ArrayT& array)
{
    auto view = array.view();
    for(int i = 0; i < view.rows(); i++) {
        for(int j = 0; j < view.cols(); j++) {
            cout << " " << view(i,j);
        }
        cout << endl;
    }
    print_view(view);
}

int main()
{
    Array2D<float,std::vector> array(3,3,9);
    array(0,0) = 0; array(0,1) = 1; array(0,2) = 2;
    array(1,0) = 3; array(1,1) = 4; array(1,2) = 5;
    array(2,0) = 6; array(2,1) = 7; array(2,2) = 8;

    auto view = array.view();
    for(int i = 0; i < view.rows(); i++) {
        for(int j = 0; j < view.cols(); j++) {
            cout << " " << view(i,j);
        }
        cout << endl;
    }

    print_array(array);
    return 0;
}



