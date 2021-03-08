#ifndef _DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_
#define _DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_

#include <memory>
#include <vector>

namespace rtac { namespace types {

// This files defines some generic types which aims at representing a very
// crude but very general build system. The goal is to define build targets
// which will automatically (and lasily) trigger the build of their
// dependencies if needed. the original purpose was to define lasy a build for
// NVIDIA OptiX 7 objects.

// The design philosophy is that for a build target, only the build parameters
// are considerered to be part the state of the object. The build output is
// considered a temporary and merely a modification of the build parameters,
// The build output is expected to be a mutable attribute of the build target
// object. This allows to define the build method as const. The goal is that
// each build target owns const references to other build target and which it
// is depending, but is still able to trigger their build function if needed.

// Another important point is the separation of the target parameters in two
// types : the build parameters and the interface parameters. A change in the
// build parameters shall trigger a rebuild of the target and the targets which
// depends on this target will be rebuild as well. A change on the interface
// parameters does not trigger a rebuild of the target itself, but should
// trigger the build of the targets which depends on the first. Note that this
// behavior should be transparent for the child targets.

template <typename TargetT,
          template <typename T> class PointerT = std::shared_ptr>
class BuildTargetHandle
{
    public:

    using TargetType = TargetT;
    using Ptr         = PointerT<TargetT>;

    protected:

    Ptr          target_;
    unsigned int version_;

    public:

    BuildTargetHandle(TargetT* target = nullptr, unsigned int version = 0);
    BuildTargetHandle(const Ptr& target, unsigned int version = 0);

    bool has_changed() const; // this checks version_ vs target_->version()
    void acknowledge();       // set this version_ to target_->version()

    operator Ptr() const;
    
    // This enables cast to point to base classes (as smart pointers would do).
    template <typename OtherTargetT>
    operator BuildTargetHandle<OtherTargetT, PointerT>() const;

    // Minimal pointer behavior
    TargetT& operator*();
    const TargetT& operator*() const;

    TargetT* get();
    const TargetT* get() const;

    TargetT* operator->();
    const TargetT* operator->() const;
    
    template <typename OtherTargetT>
    bool operator==(const OtherTargetT* other) const;
    template <typename OtherTargetT>
    bool operator==(const BuildTargetHandle<OtherTargetT>& other) const;

    operator bool() const;
};

class BuildTarget
{
    public:

    using Ptr          = BuildTargetHandle<BuildTarget>;
    using ConstPtr     = BuildTargetHandle<const BuildTarget>;
    using Dependencies = std::vector<ConstPtr>;
    struct CircularDependencyError : std::runtime_error {
        CircularDependencyError() : std::runtime_error(
            "Adding the dependency would create an infinite loop") {}
    };


    protected:
    
    // needsBuild_ and version_ are not part of the observable state of the target
    // (The observable state is only the build parameters).
    mutable bool         needsBuild_;
    mutable unsigned int version_;
    mutable Dependencies dependencies_;
    
    // A const version of bump version allows for this object to request a need
    // build from other const methods. (For example if a dependency was
    // changed).
    void bump_version(bool needsRebuild = true) const;

    // to_build and clean methods are to be reimplemented in subclasses.
    virtual void do_build() const = 0;

    BuildTarget(const Dependencies& deps = Dependencies(0));

    public:
    
    bool needs_build() const;
    bool version() const;
    void bump_version(bool needsRebuild = true);

    void add_dependency(const ConstPtr& dep); 
    const Dependencies& dependencies() const;
    void check_circular_dependencies(const ConstPtr& dep) const;

    virtual void build() const;
    virtual void clean() const {} // no cleanup by default;
};

// BuildTargetHandle implementation ////////////////////////////////////////
template <typename TargetT, template <typename T> class PointerT>
BuildTargetHandle<TargetT,PointerT>::BuildTargetHandle(TargetT* target,
                                                       unsigned int version) :
    target_(target),
    version_(version)
{}

template <typename TargetT, template <typename T> class PointerT>
BuildTargetHandle<TargetT,PointerT>::BuildTargetHandle(const Ptr& target,
                                                       unsigned int version) :
    target_(target),
    version_(version)
{}

template <typename TargetT, template <typename T> class PointerT>
bool BuildTargetHandle<TargetT,PointerT>::has_changed() const
{
    return version_ != target_->version();
}

template <typename TargetT, template <typename T> class PointerT>
void BuildTargetHandle<TargetT,PointerT>::acknowledge()
{
    version_ = target_->version();
}

template <typename TargetT, template <typename T> class PointerT>
BuildTargetHandle<TargetT,PointerT>::operator
BuildTargetHandle<TargetT,PointerT>::Ptr() const
{
    return target_;
}

template <typename TargetT, template <typename T> class PointerT>
template <typename OtherTargetT> BuildTargetHandle<TargetT,PointerT>::operator
BuildTargetHandle<OtherTargetT, PointerT>() const
{
    return BuildTargetHandle<OtherTargetT, PointerT>(target_, version_);
}

template <typename TargetT, template <typename T> class PointerT>
TargetT& BuildTargetHandle<TargetT,PointerT>::operator*()
{
    return *target_;
}

template <typename TargetT, template <typename T> class PointerT>
const TargetT& BuildTargetHandle<TargetT,PointerT>::operator*() const
{
    return *target_;
}

template <typename TargetT, template <typename T> class PointerT>
TargetT* BuildTargetHandle<TargetT,PointerT>::get()
{
    return target_.get();
}

template <typename TargetT, template <typename T> class PointerT>
const TargetT* BuildTargetHandle<TargetT,PointerT>::get() const
{
    return target_.get();
}

template <typename TargetT, template <typename T> class PointerT>
TargetT* BuildTargetHandle<TargetT,PointerT>::operator->()
{
    return this->get();
}

template <typename TargetT, template <typename T> class PointerT>
const TargetT* BuildTargetHandle<TargetT,PointerT>::operator->() const
{
    return this->get();
}

template <typename TargetT, template <typename T> class PointerT>
template <typename OtherTargetT>
bool BuildTargetHandle<TargetT,PointerT>::operator==(const OtherTargetT* other) const
{
    return ((const void*)this->get()) == ((const void*)other);
}

template <typename TargetT, template <typename T> class PointerT>
template <typename OtherTargetT>
bool BuildTargetHandle<TargetT,PointerT>::operator==(
                const BuildTargetHandle<OtherTargetT>& other) const
{
    return ((const void*)this->get()) == ((const void*)other.get());
}

template <typename TargetT, template <typename T> class PointerT>
BuildTargetHandle<TargetT,PointerT>::operator bool() const
{
    return !(!(target_));
}

// BuildTarget implementation ////////////////////////////////////////

}; //namespace types
}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_
