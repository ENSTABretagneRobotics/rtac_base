#ifndef _DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_
#define _DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_

#include <iostream>
#include <vector>

#include <rtac_base/types/Handle.h>

namespace rtac {

template <typename VectorT>
class SharedVectorBase
{
    public:

    using VectorType      = VectorT;
    using value_type      = typename VectorT::value_type;
    using VectorPtr       = Handle<VectorT>;
    using ConstVectorPtr  = Handle<const VectorT>;
    using iterator        = value_type*;
    using const_iterator  = const value_type*;

    protected:

    VectorPtr data_;

    public:

    SharedVectorBase();
    SharedVectorBase(size_t size);
    SharedVectorBase(const VectorPtr& ptr);
    SharedVectorBase(const ConstVectorPtr& ptr);
    SharedVectorBase(const SharedVectorBase& other); // parameter should not be const ?
    template <typename VectorT2>
    SharedVectorBase(const SharedVectorBase<VectorT2>& other);

    SharedVectorBase copy() const;

    SharedVectorBase& operator=(const VectorPtr& ptr);
    SharedVectorBase& operator=(const SharedVectorBase& other);
    template <typename VectorT2>
    SharedVectorBase& operator=(const SharedVectorBase<VectorT2>& other);
    
    void resize(size_t size);
    size_t size() const;

    virtual value_type* data();
    virtual const value_type* data() const;

    VectorPtr      ptr();
    ConstVectorPtr ptr() const;

    operator bool() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
};

template <typename T>
using SharedVector = SharedVectorBase<std::vector<T>>;

// Allows to check if type is a template instanciation of SharedVector
template <typename OtherT>
struct shared_vector_test { constexpr static const bool value = false; };
template <typename T>
struct shared_vector_test<SharedVectorBase<T>> { constexpr static const bool value = true; };

template <typename T>
struct ensure_shared_vector { static_assert(shared_vector_test<T>::value,
                                            "Type is not a SharedVector-derived type"); };

// implementation
template <typename VectorT>
SharedVectorBase<VectorT>::SharedVectorBase() :
    data_(NULL)
{}

template <typename VectorT>
SharedVectorBase<VectorT>::SharedVectorBase(size_t size) :
    data_(new VectorType(size))
{}

template <typename VectorT>
SharedVectorBase<VectorT>::SharedVectorBase(const VectorPtr& ptr) :
    data_(ptr)
{}

template <typename VectorT>
SharedVectorBase<VectorT>::SharedVectorBase(const ConstVectorPtr& ptr) :
    data_(new VectorType(*ptr))
{}

template <typename VectorT>
SharedVectorBase<VectorT>::SharedVectorBase(const SharedVectorBase& other) :
    data_(other.data_)
{}

template <typename VectorT> template <typename VectorT2>
SharedVectorBase<VectorT>::SharedVectorBase(const SharedVectorBase<VectorT2>& other) :
    data_(new VectorType(*(other.ptr())))
{}

template <typename VectorT>
SharedVectorBase<VectorT> SharedVectorBase<VectorT>::copy() const
{
    return SharedVectorBase(static_cast<const SharedVectorBase*>(this)->ptr());
}

template <typename VectorT>
SharedVectorBase<VectorT>& SharedVectorBase<VectorT>::operator=(const VectorPtr& ptr)
{
    data_ = ptr;
    return *this;
}

template <typename VectorT>
SharedVectorBase<VectorT>& SharedVectorBase<VectorT>::operator=(const SharedVectorBase& other)
{
    data_ = other.data_;
    return *this;
}

template <typename VectorT> template <typename VectorT2>
SharedVectorBase<VectorT>& SharedVectorBase<VectorT>::operator=(const SharedVectorBase<VectorT2>& other)
{
    *data_ = *(other.ptr());
    return *this;
}

template <typename VectorT>
void SharedVectorBase<VectorT>::resize(size_t size)
{
    data_->resize(size);
}

template <typename VectorT>
size_t SharedVectorBase<VectorT>::size() const
{
    return data_->size();
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::value_type* SharedVectorBase<VectorT>::data()
{
    return data_->data();
}

template <typename VectorT>
const typename SharedVectorBase<VectorT>::value_type* SharedVectorBase<VectorT>::data() const
{
    return data_->data();
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::VectorPtr SharedVectorBase<VectorT>::ptr()
{
    return data_;
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::ConstVectorPtr SharedVectorBase<VectorT>::ptr() const
{
    return data_;
}

template <typename VectorT>
SharedVectorBase<VectorT>::operator bool() const
{
    return data_.get() != NULL;
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::iterator SharedVectorBase<VectorT>::begin()
{
    return this->data();
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::iterator SharedVectorBase<VectorT>::end()
{
    return this->data() + this->size();
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::const_iterator SharedVectorBase<VectorT>::begin() const
{
    return this->data();
}

template <typename VectorT>
typename SharedVectorBase<VectorT>::const_iterator SharedVectorBase<VectorT>::end() const
{
    return this->data() + this->size();
}

}; //namespace rtac

template <typename VectorT>
std::ostream& operator<<(std::ostream& os, const rtac::SharedVectorBase<VectorT>& v)
{
    os << *v.ptr();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_


