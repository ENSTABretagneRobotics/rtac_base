#include <rtac_base/types/BuildTarget.h>

namespace rtac { namespace types {

    
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
        if(!dep) {
            std::cout << "Dep is null !" << std::endl << std::flush;
        }
        if(dep.has_changed()) {
            this->bump_version(true);
            return true;
            //dep.acknowledge(); // Keeping track of changes
        }
    }
    return false;
    //return needsBuild_;
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

void BuildTarget::bump_version(bool needsRebuild)
{
    if(needsRebuild) needsBuild_ = true;
    version_++;
}

void BuildTarget::add_dependency(const ConstPtr& dep)
{
    this->check_circular_dependencies(dep);
    this->dependencies_.push_back(dep);
}

const BuildTarget::Dependencies& BuildTarget::dependencies() const
{
    return dependencies_;
}

void BuildTarget::check_circular_dependencies(const ConstPtr& dep) const
{
    if(dep == this) {
        throw CircularDependencyError();
    }
    for(auto& dep : dependencies_) {
        if(dep == this) {
            throw CircularDependencyError();
        }
    }
}

void BuildTarget::build() const
{
    if(!this->needs_build())
        return;

    // cleanup before build this can be a no-op
    this->clean();

    // Building dependencies (this will build all the dependency tree).
    for(auto& dep : dependencies_) {
        if(dep->needs_build()) {
            dep->build();
        }
        dep.acknowledge();
    }
    
    this->do_build();
    this->needsBuild_ = false;
}


}; //namespace types
}; //namespace rtac
