#ifndef _DEF_RTAC_BASE_TYPES_DOMAIN_H_
#define _DEF_RTAC_BASE_TYPES_DOMAIN_H_

#include <rtac_base/types/ArrayScale.h>

namespace rtac { namespace types {

/**
 * This class represent a sampled field. It contains the samples (field values
 * at "measurements") and their coordinates.
 * 
 * The coordinate space can be of arbitrary dimension and the sampling can be
 * non-uniform.
 */
template <typename T,                        // Field element type.
          template<typename>class ArrayBaseT, // underlying Array storing the field elements
          class... FunctorTs>
class SampledField : public ArrayBaseT<T>
{
    public:
    
    using ArrayType   = ArrayBaseT<T>;
    using value_type  = T;

    using ScalerType  = ArrayScale<FunctorTs...>;
    static constexpr std::size_t Dimensionality = ScalerType::Dimentionality;
    using CoordinateScalar = typename ScalerType::value_type;
    using Coordinates      = typename ScalerType::OutputType;

    protected:

    ScalerType scalers_;
    
    public:
    
    template <typename... Args>
    SampledField(ScalerType scales, Args... args) :
        ArrayBaseT<T>(args...),
        scalers_(scales)
    {}
    
    RTAC_HOSTDEVICE const ScalerType& scalers() const { return scalers_; }
    
    template <typename... Indexes> RTAC_HOSTDEVICE
    Coordinates coordinates(Indexes... indexes) const {
        return scalers_(indexes...);
    }

    RTAC_HOSTDEVICE CoordinateScalar
    get_dimension_coordinate(std::size_t dimIdx, std::size_t idx) const {
        return scalers_.get(dimIdx, idx);
    }
};

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_DOMAIN_H_


