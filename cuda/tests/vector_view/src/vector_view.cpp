#include <iostream>
using namespace std;

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;
using namespace rtac;

#include "vector_view.h"

CudaVector<float> generate_data()
{
    std::vector<float> data(10);
    for(int i = 0; i < data.size(); i++) data[i] = i;
    return data;
}

int main()
{
    CudaVector<float> input(generate_data());
    CudaVector<float> output(input.size());

    copy(input,output);

    cout <<  input << endl;
    cout << output << endl;
    
    // does not compile (intended). CudaVector.operator[] not defined on CPU)
    // cout << VectorView(output).front() << endl;

    HostVector<float> tmp(output);
    cout << tmp.view().back() << endl;

    return 0;
}
