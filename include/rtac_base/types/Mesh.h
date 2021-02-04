#ifndef _DEF_RTAC_BASE_TYPES_MESH_H_
#define _DEF_RTAC_BASE_TYPES_MESH_H_

#include <iostream>
#include <vector>

#include <rtac_base/types/Point.h>
#include <rtac_base/happly.h>

namespace rtac { namespace types {

template <typename PointT = Point3<float>,
          typename FaceT  = Point3<uint32_t>,
          template <typename> class VectorT = std::vector>
class Mesh
{
    public:

    using Point       = PointT;
    using Face        = FaceT;
    using PointVector = VectorT<Point>;
    using FaceVector  = VectorT<Face>;
    template <typename T>
    using Vector      = VectorT<T>;

    protected:

    PointVector points_;
    FaceVector  faces_;

    public:

    Mesh();
    Mesh(size_t numPoints, size_t numFaces);
    template <template<typename> class VectorT2>
    Mesh(const Mesh<PointT,FaceT,VectorT2>& other);
    
    size_t num_points() const;
    size_t num_faces()  const;

    const PointVector& points() const;
    const FaceVector&  faces()  const;

    const Point& point(size_t idx) const;
    const Face&  face(size_t idx)  const;

    Point& point(size_t idx);
    Face&  face(size_t idx);

    // Some helpful builder functions
    static Mesh<PointT,FaceT,VectorT> cube(float scale = 1.0);

    //// .ply files
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    static Mesh<PointT,FaceT,VectorT> from_ply(const std::string& path);
    
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    void export_ply(const std::string& path, bool ascii=false) const;
};

// Implementation
template <typename PointT, typename FaceT, template <typename> class VectorT>
Mesh<PointT,FaceT,VectorT>::Mesh() :
    points_(0),
    faces_ (0)
{}

template <typename PointT, typename FaceT, template <typename> class VectorT>
Mesh<PointT,FaceT,VectorT>::Mesh(size_t numPoints, size_t numFaces) :
    points_(numPoints),
    faces_ (numFaces)
{}

template <typename PointT, typename FaceT, template <typename> class VectorT>
template <template<typename> class VectorT2>
Mesh<PointT,FaceT,VectorT>::Mesh(const Mesh<PointT,FaceT,VectorT2>& other) :
    Mesh<PointT,FaceT,VectorT>(other.num_points(), other.num_faces())
{
    points_ = other.points();
    faces_  = other.faces();
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
size_t Mesh<PointT,FaceT,VectorT>::num_points() const
{
    return points_.size();
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
size_t Mesh<PointT,FaceT,VectorT>::num_faces() const
{
    return faces_.size();
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
const typename Mesh<PointT,FaceT,VectorT>::PointVector&
    Mesh<PointT,FaceT,VectorT>::points() const
{
    return points_;
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
const typename Mesh<PointT,FaceT,VectorT>::FaceVector&
    Mesh<PointT,FaceT,VectorT>::faces() const
{
    return faces_;
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
const typename Mesh<PointT,FaceT,VectorT>::Point& 
    Mesh<PointT,FaceT,VectorT>::point(size_t idx) const
{
    return points_[idx];
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
const typename Mesh<PointT,FaceT,VectorT>::Face& 
    Mesh<PointT,FaceT,VectorT>::face(size_t idx) const
{
    return faces_[idx];
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
typename Mesh<PointT,FaceT,VectorT>::Point&
    Mesh<PointT,FaceT,VectorT>::point(size_t idx)
{
    return points_[idx];
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
typename Mesh<PointT,FaceT,VectorT>::Face&
    Mesh<PointT,FaceT,VectorT>::face(size_t idx)
{
    return faces_[idx];
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
Mesh<PointT,FaceT,VectorT> Mesh<PointT,FaceT,VectorT>::cube(float scale)
{
    Mesh<PointT,FaceT,VectorT> res(8, 12);
    res.point(0) = PointT({-scale,-scale,-scale});
    res.point(1) = PointT({ scale,-scale,-scale});
    res.point(2) = PointT({ scale, scale,-scale});
    res.point(3) = PointT({-scale, scale,-scale});
    res.point(4) = PointT({-scale,-scale, scale});
    res.point(5) = PointT({ scale,-scale, scale});
    res.point(6) = PointT({ scale, scale, scale});
    res.point(7) = PointT({-scale, scale, scale});
    res.face( 0) = FaceT({0,2,1});
    res.face( 1) = FaceT({0,3,2});
    res.face( 2) = FaceT({4,5,6});
    res.face( 3) = FaceT({4,6,7});
    res.face( 4) = FaceT({0,1,5});
    res.face( 5) = FaceT({0,5,4});
    res.face( 6) = FaceT({1,2,6});
    res.face( 7) = FaceT({1,6,5});
    res.face( 8) = FaceT({2,3,7});
    res.face( 9) = FaceT({2,7,6});
    res.face(10) = FaceT({3,0,4});
    res.face(11) = FaceT({3,4,7});
    return res;
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
template <typename PointScalarT, typename FaceIndexT>
Mesh<PointT,FaceT,VectorT> Mesh<PointT,FaceT,VectorT>::from_ply(const std::string& path)
{
    happly::PLYData data(path);

    std::vector<PointScalarT> px = data.getElement("vertex").getProperty<PointScalarT>("x");
    std::vector<PointScalarT> py = data.getElement("vertex").getProperty<PointScalarT>("y");
    std::vector<PointScalarT> pz = data.getElement("vertex").getProperty<PointScalarT>("z");

    std::vector<std::string> names({"vertex_indices", "vertex_index"});
    std::vector<std::vector<FaceIndexT>> f;
    for(auto& name : names) {
        try {
            f = data.getElement("face").getListPropertyAnySign<FaceIndexT>(name);
            break;
        }
        catch(const std::runtime_error& e) {
            // wrong face index name, trying another
        }
    }
    
    Mesh<PointT,FaceT,VectorT> res(px.size(), f.size());
    for(int i = 0; i < px.size(); i++) {
        res.point(i).x = px[i];
        res.point(i).y = py[i];
        res.point(i).z = pz[i];
    }
    for(int i = 0; i < f.size(); i++) {
        res.face(i).x = f[i][0];
        res.face(i).y = f[i][1];
        res.face(i).z = f[i][2];
    }
    
    return res;
}

template <typename PointT, typename FaceT, template <typename> class VectorT>
template <typename PointScalarT, typename FaceIndexT>
void Mesh<PointT,FaceT,VectorT>::export_ply(const std::string& path, bool ascii) const
{
    happly::PLYData data;
    
    if(this->num_points() <= 0)
        return;
    data.addElement("vertex", this->num_points());
    auto& vElement = data.getElement("vertex");
    
    // have to do a copy because of the way happly is implemented
    std::vector<PointScalarT> x(this->num_points());
    std::vector<PointScalarT> y(this->num_points());
    std::vector<PointScalarT> z(this->num_points());
    for(int i = 0; i < this->num_points(); i++) {
        x[i] = this->point(i).x;
        y[i] = this->point(i).y;
        z[i] = this->point(i).z;
    }
    vElement.addProperty("x", x);
    vElement.addProperty("y", y);
    vElement.addProperty("z", z);
    
    if(this->num_faces() > 0) {
        data.addElement("face", this->num_faces());
        // ugly. See alternatives to happly.
        std::vector<std::vector<FaceIndexT>> faces(this->num_faces());
        for(size_t i = 0; i < faces.size(); i++) {
            faces[i].resize(3);
            faces[i][0] = this->face(i).x;
            faces[i][1] = this->face(i).y;
            faces[i][2] = this->face(i).z;
        }
        data.getElement("face").addListProperty("vertex_indices", faces);
    }
    
    if(ascii)
        data.write(path, happly::DataFormat::ASCII);
    else
        data.write(path, happly::DataFormat::Binary);
}


}; //namespace types
}; //namespace rtac

template <typename PointT, typename FaceT, template <typename> class VectorT>
std::ostream& operator<<(std::ostream& os,
                         const rtac::types::Mesh<PointT,FaceT,VectorT>& mesh)
{
    const char* prefix = "\n    ";

    os << "Mesh : (" << mesh.num_points() << " points, "
       << mesh.num_faces() << " faces)\n";

    os << "- Points :";
    if(mesh.num_points() < 16) {
        for(auto& p : mesh.points()) {
            os << prefix << p;
        }
    }
    else {
        auto it = mesh.points().begin();
        for(auto end = mesh.points().begin() + 3; it != end; it++) {
            os << prefix << *it;
        }
        os << prefix << "...";
        it = mesh.points().end() - 3;
        for(auto end = mesh.points().end(); it != end; it++) {
            os << prefix << *it;
        }
    }
    os << "\n- Faces :";
    if(mesh.num_faces() < 16) {
        for(auto& p : mesh.faces()) {
            os << prefix << p;
        }
    }
    else {
        auto it = mesh.faces().begin();
        for(auto end = mesh.faces().begin() + 3; it != end; it++) {
            os << prefix << *it;
        }
        os << prefix << "...";
        it = mesh.faces().end() - 3;
        for(auto end = mesh.faces().end(); it != end; it++) {
            os << prefix << *it;
        }
    }
    
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_MESH_H_




