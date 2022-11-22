#include <rtac_base/types/NDPoint.h>
using namespace rtac;

void add_ndpoint(NDPoint<float,4>& p0, const NDPoint<float,4>& p1)
{
    p0 += p1;
}

void add_arrays(float p0[4], float p1[4])
{
    p0[0] += p1[0];
    p0[1] += p1[1];
    p0[2] += p1[2];
    p0[3] += p1[3];
}
