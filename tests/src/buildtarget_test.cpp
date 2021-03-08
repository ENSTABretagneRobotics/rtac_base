#include <iostream>
using namespace std;

#include <rtac_base/types/BuildTarget.h>
using namespace rtac::types;

class TargetBase : public BuildTarget
{
    public:

    using Ptr     = BuildTargetHandle<TargetBase>;
    using ConsPtr = BuildTargetHandle<const TargetBase>;

    public:

    int id_;

    virtual void do_build() const {
        cout << "Building TargetBase : " << id_ << endl;
    }

    TargetBase(int id = 0) : id_(id) {}

    public:

    static Ptr Create(int id) {
        return Ptr(new TargetBase(id));
    }

    void check_dependencies() const
    {
        if(dependencies_.size() > 0) {
            cout << "Check dependencies status:\n";
            for(auto dep : dependencies_) {
                cout << "dependency " << dynamic_cast<const TargetBase*>(dep.get())->id_
                     << ", needs_build : " << dep->needs_build()
                     << ", has_changed : " << dep.has_changed() << endl;
            }
        }
        else {
            cout << "No dependencies" << endl;
        }
        cout << "needsBuild_ : " << needsBuild_ << endl;
        cout << "Needs build : " << this->needs_build() << endl;
    }
};

class TargetChild0 : public TargetBase
{
    public:

    using Ptr     = BuildTargetHandle<TargetChild0>;
    using ConsPtr = BuildTargetHandle<const TargetChild0>;

    virtual void do_build() const {
        cout << "Building TargetChild0 : " << id_ << endl;
    }

    TargetChild0(int id) : TargetBase(id) {}

    public:

    static Ptr Create(int id) {
        return Ptr(new TargetChild0(id));
    }
};

int main()
{
    auto target0 = TargetBase::Create(0);

    auto target1 = TargetChild0::Create(1);
    target1->add_dependency(target0);

    auto target2 = TargetChild0::Create(2);
    target2->add_dependency(target0);
    target2->add_dependency(target1);
    
    cout << "Full build" << endl;
    target2->check_dependencies();
    target2->build();
    cout << endl;
    
    cout << "No bumps" << endl;
    target2->check_dependencies();
    target2->build();
    cout << endl;

    cout << "Bumping target1" << endl;
    target1->bump_version();
    target2->check_dependencies();
    target2->build();
    cout << endl;

    cout << "No bumps" << endl;
    target2->check_dependencies();
    target2->build();
    cout << endl;

    cout << "Bumping target0 (no rebuild)" << endl;
    target0->bump_version(false);
    target2->check_dependencies();
    target2->build();
    cout << endl;

    cout << "No bumps" << endl;
    target2->check_dependencies();
    target2->build();
    cout << endl;

    return 0;
}




