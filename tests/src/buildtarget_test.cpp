#include <iostream>
using namespace std;

#include <rtac_base/types/BuildTarget.h>
using namespace rtac;

class TargetBase : public BuildTarget
{
    public:

    using Ptr     = std::shared_ptr<TargetBase>;
    using ConsPtr = std::shared_ptr<const TargetBase>;

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
                cout << "dependency " << dynamic_cast<const TargetBase*>(dep.target().get())->id_
                     << ", needs_build : " << dep.target()->needs_build()
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

    using Ptr     = std::shared_ptr<TargetChild0>;
    using ConsPtr = std::shared_ptr<const TargetChild0>;

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

    cout << "target0    : " << target0.get() << endl;
    cout << "target1    : " << target1.get() << endl;
    cout << "target2    : " << target2.get() << endl;

    BuildTarget::ConstPtr targetPtr0 = target0;
    BuildTarget::ConstPtr targetPtr1 = target1;
    BuildTarget::ConstPtr targetPtr2 = target2;

    cout << "targetPtr0 : " << targetPtr0.get() << endl;
    cout << "targetPtr1 : " << targetPtr1.get() << endl;
    cout << "targetPtr2 : " << targetPtr2.get() << endl;

    for(auto dep : target2->dependencies()) {
        cout << "dep     : " << dep.target().get() << endl;
    }
    cout << endl;
    
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




