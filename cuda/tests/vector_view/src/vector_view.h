#pragma once

#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;
#include <rtac_base/types/VectorView.h>
using namespace rtac::types;



void copy(const DeviceVector<float>& input, DeviceVector<float>& output);
