#ifndef _DEF_RTAC_BASE_TYPES_DEVICE_DUAL_MESH_H_
#define _DEF_RTAC_BASE_TYPES_DEVICE_DUAL_MESH_H_

#include <rtac_base/types/DualMesh.h>
#include <rtac_base/cuda/CudaVector.h>
#include <rtac_base/cuda/DeviceMesh.h>

namespace rtac {

//template <typename MeshT> using DualMesh = rtac::DualMesh<MeshT>;

/**
 * The DualMesh of a Mesh is a mesh where triangles are defined as triplet of
 * edges instead of triplets of vertices. They are usefull for some
 * morphological operations.
 */
template <>
class DualMesh<cuda::DeviceMesh<>>
{
    public:

    using MeshType = cuda::DeviceMesh<>;
    using Point    = typename MeshType::Point;
    using Face     = typename MeshType::Face;

    protected:

    cuda::CudaVector<Point> points_;
    cuda::CudaVector<Edge>  edges_;
    cuda::CudaVector<Face>  faces_; // these are

    void reserve(unsigned int pointCount, 
                 unsigned int faceCount,
                 unsigned int edgeCount,
                 unsigned int level);

    void subdivide(unsigned int pointCount, unsigned int faceCount,
                   unsigned int edgeCount, unsigned int level);
    
    public:

    DualMesh(const cuda::DeviceMesh<>& mesh, unsigned int subdivide = 0);

    cuda::DeviceMesh<>::Ptr create_mesh();

    const cuda::CudaVector<Point>& points() const { return points_; }
    const cuda::CudaVector<Edge>&  edges()  const { return edges_;  }
    const cuda::CudaVector<Face>&  faces()  const { return faces_;  }
};

namespace cuda {

using DeviceDualMesh = rtac::DualMesh<cuda::DeviceMesh<>>;

};

}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os,
                                const rtac::DualMesh<rtac::cuda::DeviceMesh<>>& mesh)
{
    os << "DeviceDualMesh :"
       << "\n- points : " << mesh.points().size()
       << "\n- edges  : " << mesh.edges().size()
       << "\n- faces  : " << mesh.faces().size();

    return os;
}


#endif //_DEF_RTAC_BASE_TYPES_DEVICE_DUAL_MESH_H_
