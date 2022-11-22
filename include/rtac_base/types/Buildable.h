#ifndef _DEF_RTAC_BASE_BUILDABLE_H_
#define _DEF_RTAC_BASE_BUILDABLE_H_

#include <memory>
#include <type_traits>

namespace rtac {

struct Buildable
{
    // Abstract class representing a buildable object. The BuildableHandle
    // being templated, buildables do not have to explicitly derive from this
    // interface but must implement it nonetheless.

    virtual bool needs_build() const = 0;
    virtual unsigned int build_number() const = 0;

    // // The build function is const because only the build parameters are
    // // considered part of the observable state of the buildable object.  (the
    // // build ouput is expected to be a mutable attribute).
    // virtual void build() const = 0;
};

template <typename BuildableT,
          template <typename T> class PointerT = std::shared_ptr>
class BuildableHandle
{
    public:
    
    using BuildableType = BuildableT;
    using Ptr           = PointerT<BuildableType>;

    protected:
    
    Ptr buildable_;
    unsigned int currentBuildNumber_;

    public:

    BuildableHandle(BuildableT* buildable = nullptr,
                    unsigned int buildNumber = 0);
    BuildableHandle(const PointerT<BuildableType>& buildable,
                    unsigned int buildNumber = 0);

    bool was_updated() const;
    
    operator Ptr() const;

    // This allows for casting to other types of Buildable handles (as smart
    // pointers are castable into one other on certain conditions, const
    // qualifiers, inheritance...). This works only if an implicit smart
    // pointer cast works too, otherwise an error will be generated at compile
    // time.
    template <typename OtherBuildableT>
    operator BuildableHandle<OtherBuildableT, PointerT>() const;
    
    // Minimal pointer behavior
    BuildableT& operator*();
    const BuildableT& operator*() const;

    BuildableT* operator->();
    const BuildableT* operator->() const;

    operator bool() const;
};

template <typename BuildableT, template <typename T> class PointerT>
BuildableHandle<BuildableT,PointerT>::BuildableHandle(
        BuildableT* buildable, unsigned int buildNumber) :
    buildable_(buildable),
    currentBuildNumber_(buildNumber)
{}

template <typename BuildableT, template <typename T> class PointerT>
BuildableHandle<BuildableT,PointerT>::BuildableHandle(
        const PointerT<BuildableType>& buildable, unsigned int buildNumber) :
    buildable_(buildable),
    currentBuildNumber_(buildNumber)
{}

template <typename BuildableT, template <typename T> class PointerT>
bool BuildableHandle<BuildableT,PointerT>::was_updated() const
{
    return (currentBuildNumber_ != buildable_->build_number())
            || buildable_->needs_build();
}

template <typename BuildableT, template <typename T> class PointerT>
BuildableHandle<BuildableT,PointerT>::operator
BuildableHandle<BuildableT,PointerT>::Ptr() const
{
    return buildable_;
}

template <typename BuildableT, template <typename T> class PointerT>
template <typename OtherBuildableT> BuildableHandle<BuildableT,PointerT>::operator
BuildableHandle<OtherBuildableT, PointerT>() const
{
    return BuildableHandle<OtherBuildableT, PointerT>(buildable_, currentBuildNumber_);
}

template <typename BuildableT, template <typename T> class PointerT>
BuildableT& BuildableHandle<BuildableT,PointerT>::operator*()
{
    return *buildable_;
}

template <typename BuildableT, template <typename T> class PointerT>
const BuildableT& BuildableHandle<BuildableT,PointerT>::operator*() const
{
    return *buildable_;
}

template <typename BuildableT, template <typename T> class PointerT>
BuildableT* BuildableHandle<BuildableT,PointerT>::operator->()
{
    return buildable_.get();
}

template <typename BuildableT, template <typename T> class PointerT>
const BuildableT* BuildableHandle<BuildableT,PointerT>::operator->() const
{
    return buildable_.get();
}

template <typename BuildableT, template <typename T> class PointerT>
BuildableHandle<BuildableT,PointerT>::operator bool() const
{
    return !(!(buildable_));
}


}; //namespace rtac

#endif //_DEF_RTAC_BASE_BUILDABLE_H_
