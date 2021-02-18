#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/MappedPointer.h>

class Resource
{
    public:

    using value_type = float;

    using MappedPointer      = rtac::types::MappedPointer<Resource>;
    using ConstMappedPointer = rtac::types::MappedPointer<const Resource>;

    friend MappedPointer;

    std::vector<float> data_;

    size_t size() const { return data_.size(); }

    float* do_map() {
        cout << "Mapping action" << endl;
        return data_.data();
    }

    const float* do_map() const {
        cout << "Const mapping action" << endl;
        return data_.data();
    }

    Resource(int size) : data_(size) {
        for(int i = 0; i < size; i++) data_[i] = i;
    }

    void unmap() const { cout << "Unmapping action" << endl; }

    MappedPointer map() {
        return MappedPointer(this, &Resource::do_map, &Resource::unmap);
    }

    ConstMappedPointer map() const {
        return ConstMappedPointer(this, &Resource::do_map, &Resource::unmap);
    }
};
std::ostream& operator<<(std::ostream& os, const Resource& res);



int main()
{
    Resource res(10);
    cout << res << endl;
    
    {
        cout << "Before mapping" << endl;
        auto p = res.map();
        for(int i = 0; i < 10; i++) {
            cout << " " << p[i];
        }
        cout << endl;
    }
    cout << "After mapping" << endl;

    return 0;
}

std::ostream& operator<<(std::ostream& os, const Resource& res)
{
    auto data = res.map();
    for(int i = 0; i < res.size(); i++)
        os << " " << data[i];
    return os;
}
