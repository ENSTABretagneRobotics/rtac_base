#pragma once

#include <rtac_base/cuda/utils.h>
#include <rtac_base/types/VectorView.h>
#include <rtac_base/cuda/VectorFloatND.h>

using namespace rtac::cuda;

void fill(VectorFloatNDView<float3> vect);

void update(VectorFloatNDView<float3> vect);
void update(rtac::types::VectorView<float3> vect);
void update(VectorFloatNDView<float4> vect);
void update(rtac::types::VectorView<float4> vect);
