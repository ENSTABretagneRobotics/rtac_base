#ifndef _DEF_RTAC_BASE_TYPES_EDGE_H_
#define _DEF_RTAC_BASE_TYPES_EDGE_H_

#include <iostream>

#include <rtac_base/cuda_defines.h>
#include <rtac_base/containers/HostVector.h>

namespace rtac {

struct Edge
{
    static constexpr unsigned int NoIndex = 0xffffffff;

    unsigned int first;
    unsigned int second;
    unsigned int index;

    RTAC_HOSTDEVICE Edge() = default;
    RTAC_HOSTDEVICE Edge(const Edge&) = default;
    RTAC_HOSTDEVICE Edge& operator=(const Edge&) = default;

    RTAC_HOSTDEVICE Edge(unsigned int f, unsigned int s,
                         unsigned int idx = NoIndex) :
        first(f), second(s), index(idx)
    {}
    
    RTAC_HOSTDEVICE bool operator==(const Edge& other) const {
        return first == other.first  && second == other.second
            || first == other.second && second == other.first;
    }
    RTAC_HOSTDEVICE bool is_inverse(const Edge& other) const {
        return first == other.second && second == other.first;
    }
    RTAC_HOSTDEVICE bool is_adjacent(const Edge& other) const {
        return first == other.first  || second == other.second
            || first == other.second || second == other.first;
    }
};

class EdgeSet
{
    protected:

    std::vector<Edge> data_;

    public:

    EdgeSet() = default;

    const Edge*  data() const { return data_.data(); }
    unsigned int size() const { return data_.size(); }

    Edge&       operator[](unsigned int idx)       { return data_[idx]; }
    const Edge& operator[](unsigned int idx) const { return data_[idx]; }

    unsigned int insert(const Edge& edge)
    {
        unsigned int i = 0;
        for(; i < data_.size(); i++) {
            if(data_[i] == edge)
                return i;
        }
        data_.push_back(edge);
        data_.back().index = i;
        return i;
    }
};


} //namespace rtac

inline std::ostream& operator<<(std::ostream& os, const rtac::Edge& edge)
{
    os << '(' << edge.first << ' ' << edge.second << ", " << edge.index << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_EDGE_H_
