#ifndef _DEF_RTAC_CUDA_DEVICE_MESH_H_
#define _DEF_RTAC_CUDA_DEVICE_MESH_H_

#include <rtac_base/types/Mesh.h>
#include <rtac_base/types/Point.h>

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>

namespace rtac { namespace cuda {

template <typename PointT = rtac::types::Point3<float>,
          typename FaceT  = rtac::types::Point3<uint32_t>>
class DeviceMesh : public rtac::types::Mesh<PointT, FaceT, DeviceVector>
{
    public:

    using Point       = PointT;
    using Face        = FaceT;
    using PointVector = DeviceVector<Point>;
    using FaceVector  = DeviceVector<Face>;
    template <typename T>
    using Vector      = DeviceVector<T>;
    using MeshBase    = rtac::types::Mesh<Point, Face, Vector>;

    DeviceMesh();
    DeviceMesh(size_t numPoints, size_t numFaces);
    template< template<typename> class VectorT>
    DeviceMesh(const rtac::types::Mesh<PointT,FaceT,VectorT>& other);

    // Some helpful builder functions
    static DeviceMesh<PointT,FaceT> cube(float scale = 1.0);

    //// .ply files
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    static DeviceMesh<PointT,FaceT> from_ply(const std::string& path);
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    void export_ply(const std::string& path, bool ascii=false) const;
};

// Implementation
template <typename PointT, typename FaceT>
DeviceMesh<PointT, FaceT>::DeviceMesh() :
    MeshBase()
{}

template <typename PointT, typename FaceT>
DeviceMesh<PointT, FaceT>::DeviceMesh(size_t numPoints, size_t numFaces) :
    MeshBase(numPoints, numFaces)
{}

template <typename PointT, typename FaceT>
template <template<typename> class VectorT>
DeviceMesh<PointT,FaceT>::DeviceMesh(const rtac::types::Mesh<PointT,FaceT,VectorT>& other) :
    DeviceMesh<PointT,FaceT>(other.num_points(), other.num_faces())
{
    this->points_ = other.points();
    this->faces_  = other.faces();
}

template <typename PointT, typename FaceT>
DeviceMesh<PointT,FaceT> DeviceMesh<PointT, FaceT>::cube(float scale)
{
    auto hostMesh = rtac::types::Mesh<PointT,FaceT,std::vector>::cube();
    return DeviceMesh<PointT,FaceT>(hostMesh);
}

template <typename PointT, typename FaceT>
template <typename PointScalarT, typename FaceIndexT>
DeviceMesh<PointT,FaceT> DeviceMesh<PointT, FaceT>::from_ply(const std::string& path)
{
    auto tmp = rtac::types::Mesh<PointT,FaceT,std::vector>::template
        from_ply<PointScalarT,FaceIndexT>(path);
    return DeviceMesh<PointT,FaceT>(tmp);
}

template <typename PointT, typename FaceT>
template <typename PointScalarT, typename FaceIndexT>
void DeviceMesh<PointT,FaceT>::export_ply(const std::string& path, bool ascii) const
{
    auto tmp = rtac::types::Mesh<PointT,FaceT,HostVector>(*this);
    tmp.template export_ply<PointScalarT,FaceIndexT>(path, ascii);
}

}; //namespace cuda
}; //namespace rtac

template <typename PointT, typename FaceT>
std::ostream& operator<<(std::ostream& os, const rtac::cuda::DeviceMesh<PointT, FaceT>& mesh)
{
    rtac::types::Mesh<PointT, FaceT, rtac::cuda::HostVector> tmp(mesh);
    os << "Device" << tmp;
    return os;
}

#endif //_DEF_RTAC_CUDA_DEVICE_MESH_H_
