#ifndef _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_
#define _DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_

#include <memory>

#include <rtac_base/types/Handle.h>

namespace rtac {

template <class VectorT>
class MappedPointerImpl
{
    public:

    using value_type      = typename VectorT::value_type;
    using MapMethodType   = value_type*(VectorT::*)(void);
    using UnmapMethodType = void(VectorT::*)(void) const;

    protected:

    VectorT*        vector_;
    value_type*     ptr_;
    UnmapMethodType unmapMethod_;

    public:

    MappedPointerImpl(VectorT* vector,
                      MapMethodType   mapMethod,
                      UnmapMethodType unmapMethod) :
        vector_(vector),
        ptr_((vector->*mapMethod)()),
        unmapMethod_(unmapMethod)
    {}
    ~MappedPointerImpl() { this->unmap(); }

    void unmap()
    {
        (vector_->*unmapMethod_)();
        vector_ = nullptr;
    }
    value_type*       ptr()       { return ptr_; }
    const value_type* ptr() const { return ptr_; }
};

template <typename VectorT>
class MappedPointer
{
    public:

    using value_type      = typename VectorT::value_type;
    using MapMethodType   = value_type*(VectorT::*)(void);
    using UnmapMethodType = void(VectorT::*)(void) const;

    protected:

    std::shared_ptr<MappedPointerImpl<VectorT>> ptr_;

    public:

    MappedPointer(VectorT*        vector,
                  MapMethodType   mapMethod,
                  UnmapMethodType unmapMethod) :
        ptr_(std::make_shared<MappedPointerImpl<VectorT>>(vector, mapMethod, unmapMethod))
    {}

    operator       value_type*()       { return ptr_->ptr(); }
    operator const value_type*() const { return ptr_->ptr(); }

    value_type&       operator[](std::size_t idx)       { return ptr_->ptr()[idx]; }
    const value_type& operator[](std::size_t idx) const { return ptr_->ptr()[idx]; }
};

template <class VectorT>
class MappedPointerImpl<const VectorT>
{
    public:

    using value_type      = typename VectorT::value_type;
    using MapMethodType   = const value_type*(VectorT::*)(void) const;
    using UnmapMethodType = void(VectorT::*)(void) const;

    protected:

    const VectorT*    vector_;
    const value_type* ptr_;
    UnmapMethodType   unmapMethod_;

    public:

    MappedPointerImpl(const VectorT*  vector,
                      MapMethodType   mapMethod,
                      UnmapMethodType unmapMethod) :
        vector_(vector),
        ptr_((vector->*mapMethod)()),
        unmapMethod_(unmapMethod)
    {}
    ~MappedPointerImpl() { this->unmap(); }

    void unmap()
    {
        (vector_->*unmapMethod_)();
        vector_ = nullptr;
    }
    const value_type* ptr() { return ptr_; }
};

template <typename VectorT>
class MappedPointer<const VectorT>
{
    public:

    using value_type      = typename VectorT::value_type;
    using MapMethodType   = const value_type*(VectorT::*)(void) const;
    using UnmapMethodType = void(VectorT::*)(void) const;

    protected:

    std::shared_ptr<MappedPointerImpl<const VectorT>> ptr_;

    public:

    MappedPointer(const VectorT*  vector,
                  MapMethodType   mapMethod,
                  UnmapMethodType unmapMethod) :
        ptr_(std::make_shared<MappedPointerImpl<const VectorT>>(vector, mapMethod, unmapMethod))
    {}

    operator const value_type*() const { return ptr_->ptr(); }
    const value_type& operator[](std::size_t idx) const { return ptr_->ptr()[idx]; }
};

}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_MAPPED_POINTER_H_
