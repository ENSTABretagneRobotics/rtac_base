#ifndef _DEF_RTAC_BASE_TYPES_DUAL_MESH_H_
#define _DEF_RTAC_BASE_TYPES_DUAL_MESH_H_

#include <iostream>

#include <rtac_base/containers/HostVector.h>
#include <rtac_base/types/Edge.h>
#include <rtac_base/types/Mesh.h>

namespace rtac {

/**
 * The DualMesh of a Mesh is a mesh where triangles are defined as triplet of
 * edges instead of triplets of vertices. They are usefull for some
 * morphological operations.
 */
template <class MeshT>
class DualMesh
{
    public:

    using MeshType = MeshT;
    using Point    = typename MeshT::Point;
    using Face     = typename MeshT::Face;

    static Face make_face(const Edge& e0, const Edge& e1, const Edge& e2);

    protected:

    HostVector<Point> points_;
    HostVector<Edge>  edges_;
    HostVector<Face>  faces_; // these are

    void reserve(unsigned int pointCount, 
                 unsigned int faceCount,
                 unsigned int edgeCount,
                 unsigned int level);

    void subdivide(unsigned int pointCount, unsigned int faceCount,
                   unsigned int edgeCount, unsigned int level);
    
    public:

    DualMesh(const MeshT& mesh, unsigned int subdivide = 0);

    typename MeshT::Ptr create_mesh();

    const HostVector<Point>& points() const { return points_; }
    const HostVector<Edge>&  edges()  const { return edges_;  }
    const HostVector<Face>&  faces()  const { return faces_;  }
};

template <class MeshT>
DualMesh<MeshT>::DualMesh(const MeshT& mesh, unsigned int subdivide) :
    faces_(pow(4,subdivide)*mesh.faces().size())
{
    HostVector<Face> meshFaces(mesh.faces());
    EdgeSet edges;
    for(unsigned int i = 0; i < meshFaces.size(); i++)
    {
        auto f = meshFaces[i];
        faces_[i].x = edges.insert(Edge(f.x, f.y));
        faces_[i].y = edges.insert(Edge(f.y, f.z));
        faces_[i].z = edges.insert(Edge(f.z, f.x));
    }

    this->reserve(mesh.points().size(),
                  mesh.faces().size(),
                  edges.size(),
                  subdivide);
    std::memcpy(points_.data(), mesh.points().data(), 
                sizeof(Point)*mesh.points().size());
    std::memcpy(edges_.data(), edges.data(),
                sizeof(Edge)*edges.size());

    this->subdivide(mesh.points().size(), mesh.faces().size(), edges.size(), subdivide);
}

template <class MeshT>
void DualMesh<MeshT>::reserve(unsigned int pointCount, 
                       unsigned int faceCount,
                       unsigned int edgeCount,
                       unsigned int level)
{
    for(unsigned int i = 0; i < level; i++) {
        pointCount += edgeCount;
        edgeCount = 2*edgeCount + 3*faceCount;
        faceCount *= 4;
    }

    points_.resize(pointCount);
    edges_.resize(edgeCount);
}

template <class MeshT>
void DualMesh<MeshT>::subdivide(unsigned int pointCount, unsigned int faceCount,
                                unsigned int edgeCount, unsigned int level)
{
    if(level == 0) {
        return;
    }
    for(unsigned int i = 0; i < edgeCount; i++) {
        Edge e = edges_[i];
        auto p0 = points_[e.first];
        auto p1 = points_[e.second];
        points_[pointCount + i] = Point{0.5f*(p0.x + p1.x),
                                        0.5f*(p0.y + p1.y),
                                        0.5f*(p0.z + p1.z)};
        edges_[i]             = Edge(e.first,  pointCount + i, i);
        edges_[edgeCount + i] = Edge(pointCount + i, e.second, edgeCount + i);
    }
    for(unsigned int i = 0; i < faceCount; i++) {
        auto& f = faces_[i];

        // indexes of all new edges that subdivide this face
        auto e0 = edges_[f.x];
        auto e1 = edges_[f.x + edgeCount];
        auto e2 = edges_[f.y];
        auto e3 = edges_[f.y + edgeCount];
        auto e4 = edges_[f.z];
        auto e5 = edges_[f.z + edgeCount];
        auto e6 = Edge(e0.second, e2.second, 2*edgeCount + 3*i);
        auto e7 = Edge(e2.second, e4.second, 2*edgeCount + 3*i + 1);
        auto e8 = Edge(e4.second, e0.second, 2*edgeCount + 3*i + 2);

        edges_[e6.index] = e6;
        edges_[e7.index] = e7;
        edges_[e8.index] = e8;
        f.x = e6.index; f.y = e7.index; f.z = e8.index;

        auto make_face = [](const Edge& e,
                            const Edge& e0, const Edge& e1,
                            const Edge& e2, const Edge& e3)
        {
            if(e0.is_adjacent(e2)) return Face{e.index, e0.index, e2.index};
            if(e0.is_adjacent(e3)) return Face{e.index, e0.index, e3.index};
            if(e1.is_adjacent(e2)) return Face{e.index, e1.index, e2.index};
            if(e1.is_adjacent(e3)) return Face{e.index, e1.index, e3.index};
            return Face{0,0,0};
        };

        unsigned int idx = faceCount + 3*i;
        faces_[idx]     = make_face(e6, e0, e1, e2, e3);
        faces_[idx + 1] = make_face(e7, e2, e3, e4, e5);
        faces_[idx + 2] = make_face(e8, e4, e5, e0, e1);
    }

    subdivide(pointCount + edgeCount, 4*faceCount, 2*edgeCount + 3*faceCount, level - 1);
}

template <class MeshT> typename
DualMesh<MeshT>::Face DualMesh<MeshT>::make_face(const Edge& e0, const Edge& e1, const Edge&)
{
    // these checks keep triangle oriented the same way
    if(e0.second == e1.first) {
        return Face{e0.first, e0.second, e1.second};
    }
    else if(e0.first == e1.first) {
        return Face{e0.second, e0.first, e1.second};
    }
    else if(e0.second == e1.second) {
        return Face{e0.first, e0.second, e1.first};
    }
    else {
        return Face{e0.second, e0.first, e1.first};
    }
}


template <class MeshT>
typename MeshT::Ptr DualMesh<MeshT>::create_mesh()
{
    auto mesh = MeshT::Create();
    HostVector<Face> faces(faces_.size());
    for(unsigned int i = 0; i < faces_.size(); i++)
    {
        faces[i] = make_face(edges_[faces_[i].x],
                             edges_[faces_[i].y],
                             edges_[faces_[i].z]);
    }

    mesh->points() = points_;
    mesh->faces()  = faces;
    return mesh;
}



}; //namespace rtac

template <class MeshT> inline
std::ostream& operator<<(std::ostream& os, const rtac::DualMesh<MeshT>& mesh)
{
    os << "DualMesh :"
       << "\n- points : " << mesh.points().size()
       << "\n- edges  : " << mesh.edges().size()
       << "\n- faces  : " << mesh.faces().size();
    return os;
}


#endif //_DEF_RTAC_BASE_TYPES_DUAL_MESH_H_
