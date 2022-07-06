#pragma once

#include <rtac_base/types/TuplePointer.h>
using namespace rtac::types;

struct Data { 
    float f1; 
    uint16_t f2; 
    size_t f3;
};
inline Data operator+(const Data& d, int offset)
{
    Data out(d);
    out.f1 += offset;
    out.f2 += offset;
    out.f3 += offset;
    return out;
}
void do_stuff(TuplePointer<float,uint16_t,size_t>& data, size_t size);
void do_stuff_array(float* p1, uint16_t* p2, size_t* p3, size_t size);
void do_stuff_struct(Data* data, size_t size);


struct Data2 { 
    double f32[4];
};
inline Data2 operator+(const Data2& d, int offset)
{
    Data2 out(d);
    out.f32[0] += offset;
    out.f32[1] += offset;
    out.f32[2] += offset;
    out.f32[3] += offset;
    return out;
}
void do_stuff(TuplePointer<double,double,double,double>& data, size_t size);
void do_stuff_array(double* f0, double* f1, double* f2, double* f3, size_t size);
void do_stuff_struct(Data2* data, size_t size);
