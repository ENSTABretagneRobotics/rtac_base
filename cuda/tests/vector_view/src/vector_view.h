#pragma once

#include <rtac_base/cuda/CudaVector.h>
using namespace rtac::cuda;
#include <rtac_base/containers/VectorView.h>
using namespace rtac;



void copy(const CudaVector<float>& input, CudaVector<float>& output);
