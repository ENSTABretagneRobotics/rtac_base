#include <iostream>
using namespace std;

#include <rtac_base/types/Buildable.h>
using namespace rtac;

class Buildable0 : public Buildable
{
    public:

    using Ptr      = BuildableHandle<Buildable0>;
    using ConstPtr = BuildableHandle<const Buildable0>;

    protected:

    bool needsBuild_;
    unsigned int buildNumber_;
    
    Buildable0() : needsBuild_(true), buildNumber_(0) {}

    public:

    static Ptr Create() { return Ptr(new Buildable0); }


    bool needs_build() const { return needsBuild_; }
    unsigned int build_number() const { return buildNumber_; }

    void build() { buildNumber_++; needsBuild_ = false; }
};

void print(const Buildable0::ConstPtr& buildable);
//void print(const Buildable0::Ptr& buildable);

int main()
{
    auto b0 = Buildable0::Create();
    print(b0);
    b0->build();
    print(b0);
    return 0;
}

void print(const Buildable0::ConstPtr& buildable)
//void print(const Buildable0::Ptr& buildable)
{
    cout << "needs_build  : " << ((buildable->needs_build()) ? "true" : "false") << endl
         << "build_number : " << buildable->build_number() << endl;
}
