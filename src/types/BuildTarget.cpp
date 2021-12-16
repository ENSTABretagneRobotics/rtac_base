#include <rtac_base/types/BuildTarget.h>

namespace rtac { namespace types {

// BuildTargetHandle implementation ////////////////////////////////////////
BuildTargetHandle::BuildTargetHandle(const TargetPtr& target) :
    target_(target),
    version_(target_->version())
{}

bool BuildTargetHandle::has_changed() const
{
    if(version_ != target_->version() || target_->needs_build())
        return true;
    return false;
}

void BuildTargetHandle::acknowledge()
{
    version_ = target_->version();
}

unsigned int BuildTargetHandle::version() const
{
    return version_;
}

BuildTargetHandle::TargetPtr BuildTargetHandle::target() const
{
    return target_;
}

// BuildTarget implementation ////////////////////////////////////////
BuildTarget::BuildTarget(const Dependencies& deps) :
    needsBuild_(true),
    version_(0),
    dependencies_(deps)
{}

bool BuildTarget::needs_build() const
{
    if(needsBuild_)
        return true;
    for(auto& dep : dependencies_) {
        if(!dep.target()) {
            std::cerr << "Dep is null !" << std::endl << std::flush;
        }
        if(dep.has_changed()) {
            this->bump_version(true);
            return true;
        }
    }
    return false;
}

unsigned int BuildTarget::version() const
{
    return version_;
}

void BuildTarget::bump_version(bool needsRebuild) const
{
    if(needsRebuild) needsBuild_ = true;
    version_++;
}

void BuildTarget::add_dependency(const ConstPtr& dep)
{
    //this->check_circular_dependencies(dep);
    if(dep->depends_on(this)) {
        throw CircularDependencyError();
    }
    this->dependencies_.push_back(BuildTargetHandle(dep));
}

const BuildTarget::Dependencies& BuildTarget::dependencies() const
{
    return dependencies_;
}

bool BuildTarget::depends_on(const BuildTarget* other) const
{
    if(other == this) {
        return true;
    }
    for(auto& dep : dependencies_) {
        if(dep.target()->depends_on(other)) {
            return true;
        }
    }
    return false;
}

void BuildTarget::build() const
{
    if(!this->needs_build())
        return;

    // cleanup before build. This can be a no-op
    this->clean();

    // Building dependencies (this will build all the dependency tree).
    for(auto& dep : dependencies_) {
        if(dep.target()->needs_build()) {
            dep.target()->build();
        }
        dep.acknowledge();
    }
    
    this->do_build();
    this->needsBuild_ = false;
}


}; //namespace types
}; //namespace rtac
