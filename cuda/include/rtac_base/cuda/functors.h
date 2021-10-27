#ifndef _DEF_RTAC_BASE_CUDA_FUNCTORS_H_
#define _DEF_RTAC_BASE_CUDA_FUNCTORS_H_

/**
 * This file implemetes various functors. The aim is to replace the operator
 * types which are less versatile.
 */
#include <tuple>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda { namespace functors {

template <typename Tout, typename Tin = Tout, typename Tscale = Tin>
struct Scaling {
    using InputT  = Tin;
    using OutputT = Tout;
    using ScaleT  = Tscale;

    Tscale scaling;

    RTAC_HOSTDEVICE Tout operator()(const Tin& input) const {
        return scaling*input;
    }
};

template <typename Tout, typename Tin = Tout, typename Toff = Tout>
struct Offset {
    using InputT  = Tin;
    using OutputT = Tout;

    Toff offset;

    RTAC_HOSTDEVICE Tout operator()(const Tin& input) const {
        return input + offset;
    }
};

}; //namespace functors
}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_FUNCTORS_H_
