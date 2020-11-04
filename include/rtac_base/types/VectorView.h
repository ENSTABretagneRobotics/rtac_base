#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <rtac_base/types/Handle.h>

namespace rtac { namespace types {

template <typename T>
class VectorView
{
    public:

    using value_type      = T;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    protected:

    pointer data_;
    size_t  size_;

    public:

    VectorView(pointer data, size_t size) : data_(data), size_(size) {}
    
    size_t size() { return size_; }

    T&        operator[](int idx)       { return data_[idx]; }
    const T&  operator[](int idx) const { return data_[idx]; }

    T*       data()       { return data_; }
    const T* data() const { return data_; }

    T*       begin()       { return data_; }
    T*       end()         { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end()   const { return data_ + size_; }
};

template <typename T>
class SharedVectorView
{
    public:

    using value_type      = T;
    using difference_type = std::ptrdiff_t;
    using reference       = value_type&;
    using const_reference = const value_type&;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using iterator        = pointer;
    using const_iterator  = const_pointer;

    protected:
    
    Handle<VectorView<T>> ptr_;

    public:

    SharedVectorView() : ptr_(NULL) {}
    SharedVectorView(const Handle<VectorView<T>>& ptr) : ptr_(ptr) {}
    SharedVectorView(const SharedVectorView& other) : ptr_(other.ptr_) {}

    size_t size()                { return ptr_->size(); }
    T& operator[](int idx)       { return (*ptr_)[idx]; }
    T  operator[](int idx) const { return (*ptr_)[idx]; }

    T* data()             { return ptr_->data(); }
    const T* data() const { return ptr_->data(); }

    T* begin() { return ptr_->begin(); }
    T* end()   { return ptr_->end();   }
    const T* begin() const { return ptr_->begin(); }
    const T* end()   const { return ptr_->end();   }
};


}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
