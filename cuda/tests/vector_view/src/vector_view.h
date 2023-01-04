#pragma once

#include <rtac_base/cuda/DeviceVector.h>
using namespace rtac::cuda;
#include <rtac_base/containers/VectorView.h>
using namespace rtac;



void copy(const DeviceVector<float>& input, DeviceVector<float>& output);
