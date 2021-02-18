#ifndef _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_
#define _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_

#include <rtac_base/types/Handle.h>

namespace rtac { namespace types {

// This type will automatically grab a resource when created and release the
// resource when going out of scope.

template <typename MappedT, typename PointerT>
class MappedPointer
{
    public:

    using MappedType  = MappedT;
    using PointerType = PointerT;

    protected:

    MappedType* mappedObject_;
    PointerType ptr_;
    void(MappedT::*unmapping_method_)(void);
    
    public:

    MappedPointer(MappedType* object,
                  PointerT(MappedT::*mapping_method)(void),
                  void(MappedT::*unmapping_method)(void)) :
        mappedObject_(object),
        ptr_((object->*mapping_method)()),
        unmapping_method_(unmapping_method)
    {}

    // disabling copying
    MappedPointer(const MappedPointer& other)            = delete;
    MappedPointer& operator=(const MappedPointer& other) = delete;

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
