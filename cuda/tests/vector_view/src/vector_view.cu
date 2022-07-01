#include "vector_view.h"
#include "vector_view.hcu"

void copy(const DeviceVector<float>& input, DeviceVector<float>& output)
{
    copy<<<1,input.size()>>>(VectorView(input), VectorView(output));
}
