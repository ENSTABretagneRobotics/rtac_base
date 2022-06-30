#ifndef _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_
#define _DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_

#include <type_traits>
#include <rtac_base/type_utils.h>

namespace rtac { namespace types {

template <typename T>
class VectorViewBase
{
    public:

    using value_type = typename std::remove_const<T>::type;

    protected:
    
    T*          data_;
    std::size_t size_;

    public:

    VectorViewBase(std::size_t size = 0, T* data = nullptr) : data_(data), size_(size) {}

    const value_type* data() const { return data_; }
    std::size_t       size() const { return size_; }
    // these can stay because no memory read is done (can be used 
    const value_type* begin() const { return data_; }
    const value_type* end()   const { return data_ + size_; }

    // // enable this only if operator[] is defined
    // value_type operator[](std::size_t idx) const { return data_[i]; }
    // value_type front() const { return data_[0]; }
    // value_type back()  const { return data_[size_ - 1]; }
};

template <typename T>
class MutableVectorView : public VectorViewBase<T>
{
    static_assert(!std::is_const<T>::value);

    public:

    using value_type = T;
    
    MutableVectorView(std::size_t size = 0, T* data = nullptr) :
        VectorViewBase<T>(size, data)
    {}

    value_type* data() { return this->data_; }
    std::size_t size() { return this->size_; }
    // these can stay because no memory read is done (can be used 
    value_type* begin() { return this->data_; }
    value_type* end()   { return this->data_ + this->size_; }
};

namespace details {

template <typename VectorT>
struct vector_view
{
    using type = typename std::conditional<std::is_const<VectorT>::value,
        VectorViewBase<const typename VectorT::value_type>,
        MutableVectorView<typename VectorT::value_type> >::type;
};

};

template <typename VectorT>
auto make_vector_view(VectorT& vector)
{
    using ViewType = typename details::vector_view<VectorT>::type;
    return ViewType(vector.size(), vector.data());
}

}; //namespace types
}; //namespace rtac

#endif  //_DEF_RTAC_BASE_TYPES_VECTOR_VIEW_H_


