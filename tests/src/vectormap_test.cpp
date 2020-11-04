#include <iostream>
#include <vector>
using namespace std;

#include <rtac_base/types/VectorView.h>
using namespace rtac::types;

template <typename T>
class Vector
{
    public:

    class Map : public VectorView<T>
    {
        protected:

        Vector& v_;

        public:

        Map(Vector& v, T* data) : VectorView<T>(data, v.size()), v_(v) {}
        ~Map() { v_.unmap(); }
    };

    protected:

    std::vector<T> data_;
    WeakHandle<Map> map_;

    public:

    Vector(size_t size) : data_(size) {}
    
    size_t size() const { return data_.size(); }

    T& operator[](int idx)             { return data_[idx]; }
    const T& operator[](int idx) const { return data_[idx]; }

    SharedVectorView<T> map()
    {
        auto ptr = map_.lock();
        if(ptr) {
            cout << "already mapped" << endl;
            return SharedVectorView<T>(ptr);
        }
        else {
            cout << "new mapping" << endl;
            Handle<Map> newMap(new Map(*this, data_.data()));
            map_ = newMap;
            return SharedVectorView<T>(newMap);
        }
    }

    void unmap() { cout << "unmapping" << endl; }
};

int main()
{
    Vector<float> v0(10);
    for(int i = 0; i < 10; i++) v0[i] = i;
    for(int i = 0; i < 10; i++) cout << v0[i] << " "; cout << endl;

    {
        cout << "Entered new scope" << endl;
        auto m = v0.map();
        for(int i = 0; i < m.size(); i++) cout << m[i] << " "; cout << endl;
        cout << "Leaving scope" << endl;
    }
    cout << "left scope" << endl; 
     
    return 0;
}

