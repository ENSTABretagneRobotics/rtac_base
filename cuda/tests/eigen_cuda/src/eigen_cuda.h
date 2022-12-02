#pragma once

#include <rtac_base/types/Pose.h>
#include <rtac_base/cuda/geometry.h>

Eigen::Matrix3f multiply(const Eigen::Matrix3f& lhs, const Eigen::Matrix3f& rhs);
