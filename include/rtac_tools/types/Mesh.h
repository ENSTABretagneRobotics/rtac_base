#ifndef _DEF_RTAC_TOOLS_TYPES_MESH_H_
#define _DEF_RTAC_TOOLS_TYPES_MESH_H_

#include <iostream>

#include <happly/happly.h>

#include <rtac_tools/types/common.h>

namespace rtac { namespace types {

using namespace rtac::types::indexing;

// Tp stands for pointType (usually float)
// Tf stands for faceType (usually uint32_t)
template <typename Tp = float, typename Tf = uint32_t, size_t D = 3>
class Mesh
{
    protected:

    VecArrayPtr<Tp,D> points_;
    VecArray3Ptr<Tf>  faces_;

    public:

    Mesh();
    Mesh(size_t numPoints, size_t numFaces);
    Mesh(const VecArrayPtr<Tp,D>& points, const VecArray3Ptr<Tf>& faces=NULL);
    Mesh(const VecArray<Tp,D>& points, const VecArray3<Tf>& faces);
    
    // getters
    Map<const Matrix<Tp>> points() const;
    Map<const Matrix<Tf>> faces()  const;
    VecArrayConstPtr<Tp,D> points_ptr() const;
    VecArray3ConstPtr<Tf>  faces_ptr()  const;
    size_t num_points() const;
    size_t num_faces()  const;
    
    // others
    void export_ply(const std::string& path, bool ascii=false);

    // Some helpful builder functions
    static Mesh<Tp,Tf,3> cube(Tp scale = 1.0);
    static Mesh<Tp,Tf,3> from_ply(const std::string& path);

};

// Implementation
template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh() :
    points_(NULL),
    faces_ (NULL)
{
}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh(size_t numPoints, size_t numFaces) :
    points_(VecArrayPtr<Tp,D>(new VecArray<Tp,D>(numPoints))),
    faces_ (VecArray3Ptr<Tf> (new VecArray3<Tf>(numFaces)))
{
}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh(const VecArrayPtr<Tp,D>& points, const VecArray3Ptr<Tf>& faces) :
    points_(points), faces_(faces)
{
}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,D>::Mesh(const VecArray<Tp,D>& points, const VecArray3<Tf>& faces) :
    points_(VecArrayPtr<Tp,D>(new VecArray<Tp,D>(points))),
    faces_ (VecArray3Ptr<Tf> (new VecArray3<Tf>(faces)))
{
}

template <typename Tp, typename Tf, size_t D>
Map<const Matrix<Tp>> Mesh<Tp,Tf,D>::points() const
{
    if (this->num_points() <= 0)
        return Map<const Matrix<Tp>>(NULL, 0, D);
    return Map<const Matrix<Tp>>(this->points_ptr()->data(), this->num_points(), D);
}

template <typename Tp, typename Tf, size_t D>
Map<const Matrix<Tf>> Mesh<Tp,Tf,D>::faces() const
{
    if (this->num_faces() <= 0)
        return Map<const Matrix<Tf>>(NULL, 0, 3);
    return Map<const Matrix<Tf>>(this->faces_ptr()->data(), this->num_faces(), 3);
}

template <typename Tp, typename Tf, size_t D>
VecArrayConstPtr<Tp,D> Mesh<Tp,Tf,D>::points_ptr() const
{
    return points_;
}

template <typename Tp, typename Tf, size_t D>
VecArray3ConstPtr<Tf> Mesh<Tp,Tf,D>::faces_ptr() const
{
    return faces_;
}

template <typename Tp, typename Tf, size_t D>
size_t Mesh<Tp,Tf,D>::num_points() const
{
    if(!this->points_ptr())
        return 0;
    return this->points_ptr()->rows();
}

template <typename Tp, typename Tf, size_t D>
size_t Mesh<Tp,Tf,D>::num_faces() const
{
    if(!this->faces_ptr())
        return 0;
    return this->faces_ptr()->rows();
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

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,3> Mesh<Tp,Tf,D>::cube(Tp scale)
{
    VecArrayPtr<Tp,3> points(new VecArray<Tp,3>(8));
    *points << -scale,-scale,-scale,
                scale,-scale,-scale,
                scale, scale,-scale,
               -scale, scale,-scale,
               -scale,-scale, scale,
                scale,-scale, scale,
                scale, scale, scale,
               -scale, scale, scale;
    VecArray3Ptr<Tf> faces (new VecArray3<Tf>(12));
    *faces << 0,2,1,
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
    return Mesh<Tp,Tf,3>(points, faces);
}

template <typename Tp, typename Tf, size_t D>
Mesh<Tp,Tf,3> Mesh<Tp,Tf,D>::from_ply(const std::string& path)
{
    happly::PLYData data(path);

    std::vector<Tp> x = data.getElement("vertex").getProperty<Tp>("x");
    std::vector<Tp> y = data.getElement("vertex").getProperty<Tp>("y");
    std::vector<Tp> z = data.getElement("vertex").getProperty<Tp>("z");

    VecArrayPtr<Tp,3> points(new VecArray<Tp,3>(x.size()));
    for(int i = 0; i < x.size(); i++) {
        (*points)(i,0) = x[i];
        (*points)(i,1) = y[i];
        (*points)(i,2) = z[i];
    }
    
    VecArray3Ptr<Tf> faces(NULL);
    std::vector<std::string> names({"vertex_indices", "vertex_index"});
    for(auto& name : names) {
        try {
            std::vector<std::vector<Tf>> f = data.getElement("face")
                .getListPropertyAnySign<Tf>(name);
            faces = VecArray3Ptr<Tf>(new VecArray3<Tf>(f.size()));
            for(int i = 0; i < f.size(); i++) {
                (*faces)(i,0) = f[i][0];
                (*faces)(i,1) = f[i][1];
                (*faces)(i,2) = f[i][2];
            }
            break;
        }
        catch(const std::runtime_error& e) {
            // wrong face index name, trying another
        }
    }

    return Mesh(points, faces);
}


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

#endif //_DEF_RTAC_TOOLS_TYPES_MESH_H_




