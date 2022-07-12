#ifndef _DEF_RTAC_BASE_TYPES_MAPPED_GRID_H_
#define _DEF_RTAC_BASE_TYPES_MAPPED_GRID_H_

#include <rtac_base/types/GridMap.h>

namespace rtac { namespace types {

/**
 * This class is an interface for a multidimensional array associated with a
 * mapping which maps the array indices to physical coordinates.
 *
 * For exemple it can be used to hold a planisphere representation. The array
 * part contains the geographically projected 2D map image and the mapping
 * encodes the projection from the map image to a sphere. So to each pixel of
 * the map image corresponds a lat,lon coordinate.
 *
 * The image pixel type is T. The ArrayBaseT<T> is the container type for the
 * data and the mappings are encoded in a set of functors.
 */
template <typename T,
          template<typename>class ArrayBaseT,
          class... FunctorTs>
class MappedGrid : public ArrayBaseT<T>
{
    public:
    
    using ArrayType   = ArrayBaseT<T>;
    using value_type  = T;

    using MappingType      = GridMap<FunctorTs...>;
    static constexpr std::size_t Dimensionality = MappingType::Dimentionality;
    using CoordinateScalar = typename MappingType::value_type;
    using Coordinates      = typename MappingType::OutputType;

    protected:

    MappingType mapping_;
    
    public:
    
    /**
     * Args are arguments for the ArrayBaseT constructor.
     */
    template <typename... Args>
    MappedGrid(MappingType mapping, Args... args) :
        ArrayBaseT<T>(args...),
        mapping_(mapping)
    {}
    
    RTAC_HOSTDEVICE const MappingType& mapping() const { return mapping_; }
    
    template <typename... Indexes> RTAC_HOSTDEVICE
    Coordinates coordinates(Indexes... indexes) const {
        return mapping_(indexes...);
    }

    RTAC_HOSTDEVICE CoordinateScalar
    get_dimension_coordinate(std::size_t dimIdx, std::size_t idx) const {
        return mapping_.get(dimIdx, idx);
    }
};

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_MAPPED_GRID_H_


