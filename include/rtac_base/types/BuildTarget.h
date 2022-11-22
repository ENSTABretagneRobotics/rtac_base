#ifndef _DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_
#define _DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_

#include <iostream>
#include <vector>

#include <rtac_base/types/Handle.h>

namespace rtac {

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

class BuildTarget;

class BuildTargetHandle
{
    public:

    using TargetPtr = Handle<const BuildTarget>;

    protected:

    TargetPtr    target_;
    unsigned int version_;

    public:

    BuildTargetHandle(const TargetPtr& target = nullptr);

    bool has_changed()  const; // this checks version_ vs target_->version()
    void acknowledge();        // set this version_ to target_->version()
    
    unsigned int version() const;
    TargetPtr    target()  const;
};

class BuildTarget
{
    public:

    using Ptr          = Handle<BuildTarget>;
    using ConstPtr     = Handle<const BuildTarget>;
    using Dependencies = std::vector<BuildTargetHandle>;

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

    // to_build and clean methods are to be reimplemented in subclasses.
    virtual void do_build() const = 0;

    BuildTarget(const Dependencies& deps = Dependencies(0));

    public:
    
    bool needs_build() const;
    unsigned int version() const;
    // A const version of bump version allows for this object to request a need
    // build from other const methods. (For example if a dependency was
    // changed).
    void bump_version(bool needsRebuild = true) const;

    void add_dependency(const ConstPtr& dep); 
    const Dependencies& dependencies() const;
    bool depends_on(const BuildTarget* other) const;

    virtual void build() const;
    virtual void clean() const {} // no cleanup by default;
};


}; //namespace rtac

#endif //_DEF_RTAC_BASE_TYPES_BUILD_TARGET_H_
