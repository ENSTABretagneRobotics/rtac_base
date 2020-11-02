#ifndef _DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_
#define _DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_

#include <iostream>

#include <rtac_base/types/Handle.h>

namespace rtac { namespace types {

template <typename VectorT>
class SharedVector
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

    SharedVector();
    SharedVector(size_t size);
    SharedVector(const VectorPtr& ptr);
    SharedVector(const ConstVectorPtr& ptr);
    SharedVector(const SharedVector& other); // parameter should not be const ?

    SharedVector copy() const;

    SharedVector& operator=(const VectorPtr& ptr);
    SharedVector& operator=(const SharedVector& other);
    
    void resize(size_t size);
    size_t size() const;

    value_type* data();
    const value_type* data() const;

    VectorPtr      ptr();
    ConstVectorPtr ptr() const;

    operator bool() const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
};


template <typename VectorT>
SharedVector<VectorT>::SharedVector() :
    data_(NULL)
{}

template <typename VectorT>
SharedVector<VectorT>::SharedVector(size_t size) :
    data_(new VectorType(size))
{}

template <typename VectorT>
SharedVector<VectorT>::SharedVector(const VectorPtr& ptr) :
    data_(ptr)
{}

template <typename VectorT>
SharedVector<VectorT>::SharedVector(const ConstVectorPtr& ptr) :
    data_(new VectorType(*ptr))
{}

template <typename VectorT>
SharedVector<VectorT>::SharedVector(const SharedVector& other) :
    data_(other.data_)
{}

template <typename VectorT>
SharedVector<VectorT> SharedVector<VectorT>::copy() const
{
    return SharedVector(static_cast<const SharedVector*>(this)->ptr());
}

template <typename VectorT>
SharedVector<VectorT>& SharedVector<VectorT>::operator=(const VectorPtr& ptr)
{
    data_ = ptr;
    return *this;
}

template <typename VectorT>
SharedVector<VectorT>& SharedVector<VectorT>::operator=(const SharedVector& other)
{
    data_ = other.data_;
    return *this;
}

template <typename VectorT>
void SharedVector<VectorT>::resize(size_t size)
{
    data_->resize(size);
}

template <typename VectorT>
size_t SharedVector<VectorT>::size() const
{
    return data_->size();
}

template <typename VectorT>
typename SharedVector<VectorT>::value_type* SharedVector<VectorT>::data()
{
    return data_->data();
}

template <typename VectorT>
const typename SharedVector<VectorT>::value_type* SharedVector<VectorT>::data() const
{
    return data_->data();
}

template <typename VectorT>
typename SharedVector<VectorT>::VectorPtr SharedVector<VectorT>::ptr()
{
    return data_;
}

template <typename VectorT>
typename SharedVector<VectorT>::ConstVectorPtr SharedVector<VectorT>::ptr() const
{
    return data_;
}

template <typename VectorT>
SharedVector<VectorT>::operator bool() const
{
    return data_;
}

template <typename VectorT>
typename SharedVector<VectorT>::iterator SharedVector<VectorT>::begin()
{
    return this->data();
}

template <typename VectorT>
typename SharedVector<VectorT>::iterator SharedVector<VectorT>::end()
{
    return this->data() + this->size();
}

template <typename VectorT>
typename SharedVector<VectorT>::const_iterator SharedVector<VectorT>::begin() const
{
    return this->data();
}

template <typename VectorT>
typename SharedVector<VectorT>::const_iterator SharedVector<VectorT>::end() const
{
    return this->data() + this->size();
}

}; //namespace types
}; //namespace rtac

template <typename VectorT>
std::ostream& operator<<(std::ostream& os, const rtac::types::SharedVector<VectorT>& v)
{
    auto data = v.data();
    os << "(";
    if(v.size() <= 10) {
        os << data[0];
        for(int i = 1; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    else {
        for(int i = 1; i < 3; i++) {
            os << data[i] << " ";
        }
        os << "...";
        for(int i = v.size() - 3; i < v.size(); i++) {
            os << " " << data[i];
        }
    }
    os << ")";
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_SHARED_VECTOR_H_


