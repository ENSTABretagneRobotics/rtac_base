#include "vector_view.h"
#include "vector_view.hcu"

void copy(const CudaVector<float>& input, CudaVector<float>& output)
{
    copy<<<1,input.size()>>>(input.view(), output.view());
}
