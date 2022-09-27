#ifndef _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_
#define _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_

#include <rtac_base/types/Handle.h>

namespace rtac { namespace types {

// This type will automatically grab a resource when created and release the
// resource when going out of scope.

template <typename MappedT,
          typename PointerT = typename std::conditional<std::is_const<MappedT>::value,
                                const typename MappedT::value_type*,
                                typename MappedT::value_type*>::type>
class MappedPointer
{
    public:

    using MappedType    = MappedT;
    using PointerType   = PointerT;
    using MapMethodType = typename std::conditional<std::is_const<MappedType>::value,
                                                    PointerType(MappedT::*)(void) const,
                                                    PointerType(MappedT::*)(void)>::type;
    using UnmapMethodType = void(MappedT::*)(void) const;

    protected:

    MappedType* mappedObject_;
    PointerType ptr_;
    UnmapMethodType unmapping_method_;
    
    public:

    MappedPointer(MappedType* object,
                  MapMethodType   mapping_method,
                  UnmapMethodType unmapping_method) :
        mappedObject_(object),
        ptr_((object->*mapping_method)()),
        unmapping_method_(unmapping_method)
    {}

    // disabling copying, // reenabled because cuda
    //MappedPointer(const MappedPointer& other)            = delete;
    //MappedPointer& operator=(const MappedPointer& other) = delete;

    ~MappedPointer()
    {
        // MappedResource will be released when this object goes out of scope.
        this->unmap();
    }

    void unmap()
    {
        (mappedObject_->*unmapping_method_)();
        mappedObject_ = nullptr;
    }

    // Implicit cast to mapped pointer.
    operator PointerType()
    {
        return ptr_;
    }
};

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_
