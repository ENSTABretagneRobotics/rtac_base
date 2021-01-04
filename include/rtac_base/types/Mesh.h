#ifndef _DEF_RTAC_BASE_TYPES_MESH_H_
#define _DEF_RTAC_BASE_TYPES_MESH_H_

#include <iostream>

#ifdef RTAC_BASE_PLY_FILES
#include <happly.h>
#endif

#include <rtac_base/types/common.h>

namespace rtac { namespace types {

using namespace rtac::types::indexing;

// Tp stands for pointType (usually float)
// Tf stands for faceType (usually uint32_t)
template <typename Tp = float, typename Tf = uint32_t, size_t D = 3>
class Mesh
{
    protected:

    Array<Tp,D> points_;
    Array3<Tf>  faces_;

    public:

    Mesh();
    Mesh(size_t numPoints, size_t numFaces);
    Mesh(const Array<Tp,D>& points, const Array3<Tf>& faces);
    
    Map<Matrix<Tp>> points();
    Map<Matrix<Tf>> faces();

    // getters
    Map<const Matrix<Tp>> points() const;
    Map<const Matrix<Tf>> faces()  const;
    size_t num_points() const;
    size_t num_faces()  const;

    // Some helpful builder functions
    static Mesh<Tp,Tf,3> cube(Tp scale = 1.0);

    // .ply files
#ifdef RTAC_BASE_PLY_FILES
    static Mesh<Tp,Tf,3> from_ply(const std::string& path);
    void export_ply(const std::string& path, bool ascii=false);
#endif

};

// Implementation
template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh() :
    points_(0),
    faces_ (0)
{}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh(size_t numPoints, size_t numFaces) :
    points_(numPoints),
    faces_ (numFaces)
{}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh(const Array<Tp,D>& points, const Array3<Tf>& faces) :
    points_(points),
    faces_ (faces)
{}

template <typename Tp, typename Tf, size_t D>
Map<Matrix<Tp>> Mesh<Tp,Tf,D>::points()
{
    if (this->num_points() <= 0)
        return Map<Matrix<Tp>>(NULL, 0, D);
    return Map<Matrix<Tp>>(points_.data(), this->num_points(), D);
}

template <typename Tp, typename Tf, size_t D>
Map<Matrix<Tf>> Mesh<Tp,Tf,D>::faces()
{
    if (this->num_faces() <= 0)
        return Map<Matrix<Tf>>(NULL, 0, 3);
    return Map<Matrix<Tf>>(faces_.data(), this->num_faces(), 3);
}

template <typename Tp, typename Tf, size_t D>
Map<const Matrix<Tp>> Mesh<Tp,Tf,D>::points() const
{
    if (this->num_points() <= 0)
        return Map<const Matrix<Tp>>(NULL, 0, D);
    return Map<const Matrix<Tp>>(points_.data(), this->num_points(), D);
}

template <typename Tp, typename Tf, size_t D>
Map<const Matrix<Tf>> Mesh<Tp,Tf,D>::faces() const
{
    if (this->num_faces() <= 0)
        return Map<const Matrix<Tf>>(NULL, 0, 3);
    return Map<const Matrix<Tf>>(faces_.data(), this->num_faces(), 3);
}

template <typename Tp, typename Tf, size_t D>
size_t Mesh<Tp,Tf,D>::num_points() const
{
    return points_.rows();
}

template <typename Tp, typename Tf, size_t D>
size_t Mesh<Tp,Tf,D>::num_faces() const
{
    return faces_.rows();
}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,3> Mesh<Tp,Tf,D>::cube(Tp scale)
{
    Mesh<Tp,Tf,3> res(8, 12);
    res.points() << -scale,-scale,-scale,
                     scale,-scale,-scale,
                     scale, scale,-scale,
                    -scale, scale,-scale,
                    -scale,-scale, scale,
                     scale,-scale, scale,
                     scale, scale, scale,
                    -scale, scale, scale;
    res.faces() << 0,2,1,
                   0,3,2,
                   4,5,6,
                   4,6,7,
                   0,1,5,
                   0,5,4,
                   1,2,6,
                   1,6,5,
                   2,3,7,
                   2,7,6,
                   3,0,4,
                   3,4,7;
    return res;
}

#ifdef RTAC_BASE_PLY_FILES
template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,3> Mesh<Tp,Tf,D>::from_ply(const std::string& path)
{
    happly::PLYData data(path);

    std::vector<Tp> px = data.getElement("vertex").getProperty<Tp>("x");
    std::vector<Tp> py = data.getElement("vertex").getProperty<Tp>("y");
    std::vector<Tp> pz = data.getElement("vertex").getProperty<Tp>("z");

    std::vector<std::string> names({"vertex_indices", "vertex_index"});
    std::vector<std::vector<Tf>> f;
    for(auto& name : names) {
        try {
            f = data.getElement("face").getListPropertyAnySign<Tf>(name);
            break;
        }
        catch(const std::runtime_error& e) {
            // wrong face index name, trying another
        }
    }
    
    Mesh res(px.size(), f.size());
    auto points = res.points();
    for(int i = 0; i < px.size(); i++) {
        points(i,0) = px[i];
        points(i,1) = py[i];
        points(i,2) = pz[i];
    }
    auto faces = res.faces();
    for(int i = 0; i < f.size(); i++) {
        faces(i,0) = f[i][0];
        faces(i,1) = f[i][1];
        faces(i,2) = f[i][2];
    }
    

    return res;
}

template <typename Tp, typename Tf, size_t D>
void Mesh<Tp,Tf,D>::export_ply(const std::string& path, bool ascii)
{
    happly::PLYData data;
    
    if(this->num_points() <= 0)
        return;
    data.addElement("vertex", this->num_points());
    auto& vElement = data.getElement("vertex");
    
    auto points = this->points();
    std::vector<Tp> x(points(all,0).begin(), points(all,0).end());
    std::vector<Tp> y(points(all,1).begin(), points(all,1).end());
    std::vector<Tp> z(points(all,2).begin(), points(all,2).end());
    vElement.addProperty("x", x);
    vElement.addProperty("y", y);
    vElement.addProperty("z", z);
    
    if(this->num_faces() > 0) {
        data.addElement("face", this->num_faces());
        auto faces = this->faces();
        // Have to do this. happly not accepting views.
        std::vector<std::vector<Tf>> f(this->num_faces());
        for(size_t i = 0; i < f.size(); i++) {
            f[i].assign(faces(i,all).begin(), faces(i,all).end());
        }
        data.getElement("face").addListProperty("vertex_indices", f);
    }
    
    if(ascii)
        data.write(path, happly::DataFormat::ASCII);
    else
        data.write(path, happly::DataFormat::Binary);
}
#endif //RTAC_BASE_PLY_FILES


}; //namespace types
}; //namespace rtac

template <typename Tp, typename Tf, size_t D>
std::ostream& operator<<(std::ostream& os, const rtac::types::Mesh<Tp,Tf,D>& mesh)
{
    os << "Mesh : (" << mesh.num_points() << " points, "
       << mesh.num_faces() << " faces)\n";
    
    if (mesh.num_points() < 20 && mesh.num_faces() < 20) {
       os << "Points :\n" << mesh.points() << "\n"
          << "Faces  :\n" << mesh.faces()  << "\n";
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_MESH_H_




