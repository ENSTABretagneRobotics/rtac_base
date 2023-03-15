#ifndef _DEF_RTAC_BASE_TYPES_MESH_H_
#define _DEF_RTAC_BASE_TYPES_MESH_H_

#include <iostream>
#include <vector>
#include <cmath>

#include <rtac_base/types/Point.h>
#include <rtac_base/types/PointCloud.h>
#include <rtac_base/containers/HostVector.h>

#include <rtac_base/cuda_defines.h>
#ifndef RTAC_CUDACC
#include <rtac_base/happly.h>
#endif

namespace rtac {

template <typename P = Point3<float>,
          typename F = Point3<uint32_t>,
          typename N = Point3<float>,
          typename U = Point2<float>,
          //template <typename> class V = std::vector>
          template <typename> class V = HostVector>
class Mesh
{
    public:

    using Point  = P;
    using Face   = F;
    using UV     = U;
    using Normal = N;
    template <typename T>
    using Vector = V<T>;

    using MeshType = Mesh<P,F,N,U,V>;

    using Ptr      = std::shared_ptr<Mesh<P,F,N,U,V>>;
    using ConstPtr = std::shared_ptr<const Mesh<P,F,N,U,V>>;

    protected:

    Vector<Point>  points_;
    Vector<Face>   faces_;
    Vector<Normal> normals_;
    Vector<UV>     uvs_;

    public:

    static Ptr Create()                                { return Ptr(new Mesh()); }
    static Ptr Create(MeshType&& other)                { return Ptr(new Mesh(other)); }
    template <template<typename> class Vect>
    static Ptr Create(const Mesh<P,F,N,U,Vect>& other) { return Ptr(new Mesh(other)); }

    Mesh() {}
    Mesh(Mesh<P,F,N,U,V>&& other) { *this = other; }
    MeshType& operator=(MeshType&& other);
    
    // These allows direct copies from RAM to GPU (either to CUDA or OpenGL,
    // using rtac::cuda::DeviceVector or rtac::display::GLVector respectively)
    template <template<typename>class Vect>
    Mesh(const Mesh<P,F,N,U,Vect>& other) { *this = other; }
    template <template<typename>class OtherVect>
    MeshType& operator=(const Mesh<P,F,N,U,OtherVect>& other);

    template <typename T>
    MeshType& operator=(const rtac::PointCloud<T>& pointcloud);

    Vector<Point>&  points()  { return points_;  } 
    Vector<Face>&   faces()   { return faces_;   } 
    Vector<Normal>& normals() { return normals_; } 
    Vector<UV>&     uvs()     { return uvs_;     }

    const Vector<Point>&  points()  const { return points_;  } 
    const Vector<Face>&   faces()   const { return faces_;   } 
    const Vector<Normal>& normals() const { return normals_; } 
    const Vector<UV>&     uvs()     const { return uvs_;     }

    // Some helpful builder functions
    static Ptr cube(float scale = 1.0);
    static Ptr sphere_section(float radius,
                              float xAperture,
                              float yAperture,
                              size_t Nx,
                              size_t Ny);
    //static Ptr icosphere(unsigned int level = 0, float scale = 1.0f);
    static Ptr icosahedron(float scale = 1.0f);

    //// .ply files
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    static Ptr from_ply(const std::string& path);
    
    template <typename PointScalarT = float, typename FaceIndexT = uint32_t>
    void export_ply(const std::string& path, bool ascii=false) const;
};

// Implementation
template <typename P, typename F, typename N, typename U, template<typename> class V>
Mesh<P,F,N,U,V>& Mesh<P,F,N,U,V>::operator=(Mesh<P,F,N,U,V>&& other)
{
    points_  = std::move(other.points());
    faces_   = std::move(other.faces());
    normals_ = std::move(other.normals());
    uvs_     = std::move(other.uvs());
    return *this;
}

template <typename P, typename F, typename N, typename U, template<typename> class V>
template <template <typename> class OtherVect>
Mesh<P,F,N,U,V>& Mesh<P,F,N,U,V>::operator=(const Mesh<P,F,N,U,OtherVect>& other)
{
    points_  = other.points();
    faces_   = other.faces();
    normals_ = other.normals();
    uvs_     = other.uvs();
    return *this;
}


template <typename P, typename F, typename N, typename U, template<typename> class V>
typename Mesh<P,F,N,U,V>::Ptr Mesh<P,F,N,U,V>::cube(float scale)
{
    auto res = Mesh<P,F,N,U,V>::Create();

    std::vector<Point> points(8);
    points[0] = Point({-scale,-scale,-scale});
    points[1] = Point({ scale,-scale,-scale});
    points[2] = Point({ scale, scale,-scale});
    points[3] = Point({-scale, scale,-scale});
    points[4] = Point({-scale,-scale, scale});
    points[5] = Point({ scale,-scale, scale});
    points[6] = Point({ scale, scale, scale});
    points[7] = Point({-scale, scale, scale});
    res->points() = points;

    std::vector<Face> faces(12);
    faces[ 0] = Face({0,3,2});
    faces[ 1] = Face({0,2,1});
    faces[ 2] = Face({4,5,6});
    faces[ 3] = Face({4,6,7});
    faces[ 4] = Face({0,1,5});
    faces[ 5] = Face({0,5,4});
    faces[ 6] = Face({1,2,6});
    faces[ 7] = Face({1,6,5});
    faces[ 8] = Face({2,3,7});
    faces[ 9] = Face({2,7,6});
    faces[10] = Face({3,0,4});
    faces[11] = Face({3,4,7});
    res->faces()  = faces;

    return res;
}

template <typename P, typename F, typename N, typename U, template<typename> class V>
typename Mesh<P,F,N,U,V>::Ptr Mesh<P,F,N,U,V>::sphere_section(float  radius,
                                                              float  xAperture,
                                                              float  yAperture,
                                                              size_t Nx,
                                                              size_t Ny)
{
    auto res = Mesh<P,F,N,U,V>::Create();

    std::vector<Point> points(Nx*Ny + 1);
                       //2*(Nx*Ny - 1));
    points[0] = Point({0,0,0});
    for(int ny = 0; ny < Ny; ny++) {
        auto cy = radius*cos(yAperture*(((float)ny) / (Ny - 1) - 0.5f));
        auto sy = radius*sin(yAperture*(((float)ny) / (Ny - 1) - 0.5f));
        for(int nx = 0; nx < Nx; nx++) {
            auto cx = cos(xAperture*(((float)nx) / (Nx - 1) - 0.5f));
            auto sx = sin(xAperture*(((float)nx) / (Nx - 1) - 0.5f));
            points[Nx*ny + nx + 1] = Point({cx*cy, sx*cy, sy});
        }
    }
    res->points() = points;
    
    // size_t nf = 0;
    // for(int nx = 0; nx < Nx - 1; nx++) {
    //     res.face(nf) = FaceT({0, nx+2, nx+1});
    //     nf++;
    // }
    // for(int ny = 0; ny < Ny-1; ny++) {
    //     res.face(nf) = FaceT({0, Nx*ny+1, Nx*(ny - 1)+1});
    //     nf++;
    // }
    // for(int nx = 0; nx < Nx - 1; nx++) {
    //     res.face(nf) = FaceT({0, Nx*Ny-nx-1, Nx*Ny-nx});
    //     nf++;
    // }
    // for(int ny = 0; ny < Ny-1; ny++) {
    //     res.face(nf) = FaceT({0, Nx*(Ny-1) - Nx*(ny+1) + 1, Nx*(Ny-1) - Nx*ny + 1});
    //     nf++;
    // }

    // for(int ny = 0; ny < Ny - 2; ny++) {
    //     for(int nx = 0; nx < Nx - 2; nx++) {
    //         unsigned int Nc = Nx*ny + nx + 1;
    //         res.face(nf)   = FaceT({Nc, Nc+1,  Nc+1+Nx});
    //         res.face(nf+1) = FaceT({Nc, Nc+1+Nx, Nc+Nx});
    //         nf += 2;
    //     }
    // }
    return res;
}

template <typename P, typename F, typename N, typename U, template<typename> class V>
typename Mesh<P,F,N,U,V>::Ptr Mesh<P,F,N,U,V>::icosahedron(float scale)
{
    auto res = Mesh<P,F,N,U,V>::Create();

    constexpr float phi  = 0.5f*(1.0f + sqrt(5));
    scale /= sqrt(1 + phi*phi);

    std::vector<P> points(12);
    points[0]  = P{-phi*scale, -1.0f*scale, 0.0f};
    points[1]  = P{ phi*scale, -1.0f*scale, 0.0f};
    points[2]  = P{ phi*scale,  1.0f*scale, 0.0f};
    points[3]  = P{-phi*scale,  1.0f*scale, 0.0f};

    points[4]  = P{-1.0f*scale, 0.0f, -phi*scale};
    points[5]  = P{-1.0f*scale, 0.0f,  phi*scale};
    points[6]  = P{ 1.0f*scale, 0.0f,  phi*scale};
    points[7]  = P{ 1.0f*scale, 0.0f, -phi*scale};

    points[8]  = P{0.0f, -phi*scale, -1.0f*scale};
    points[9]  = P{0.0f,  phi*scale, -1.0f*scale};
    points[10] = P{0.0f,  phi*scale,  1.0f*scale};
    points[11] = P{0.0f, -phi*scale,  1.0f*scale};

    std::vector<F> faces(20);
    faces[0]  = F{1,2,6};
    faces[1]  = F{2,1,7};
    faces[2]  = F{3,0,5};
    faces[3]  = F{0,3,4};

    faces[4]  = F{5,6,10};
    faces[5]  = F{6,5,11};
    faces[6]  = F{7,4,9};
    faces[7]  = F{4,7,8};

    faces[8]  = F{9,10,2};
    faces[9]  = F{10,9,3};
    faces[10] = F{11,8,1};
    faces[11] = F{8,11,0};

    faces[12] = F{0,11,5};
    faces[13] = F{1,6,11};
    faces[14] = F{2,10,6};
    faces[15] = F{3,5,10};

    faces[16] = F{0,4,8};
    faces[17] = F{1,8,7};
    faces[18] = F{2,7,9};
    faces[19] = F{3,9,4};

    res->points() = points;
    res->faces()  = faces;

    return res;
}

#ifndef RTAC_CUDACC
template <typename P, typename F, typename N, typename U, template<typename> class V>
template <typename PointScalarT, typename FaceIndexT>
typename Mesh<P,F,N,U,V>::Ptr Mesh<P,F,N,U,V>::from_ply(const std::string& path)
{
    auto res = MeshType::Create();

    happly::PLYData data(path);
    {
        std::vector<PointScalarT> px = data.getElement("vertex").getProperty<PointScalarT>("x");
        std::vector<PointScalarT> py = data.getElement("vertex").getProperty<PointScalarT>("y");
        std::vector<PointScalarT> pz = data.getElement("vertex").getProperty<PointScalarT>("z");
        
        // This wastes memory but has a greater flexibility
        std::vector<Point> points(px.size());
        for(int i = 0; i < px.size(); i++) {
            points[i].x = px[i];
            points[i].y = py[i];
            points[i].z = pz[i];
        }
        res->points() = points;
    }

    {
        std::vector<std::string> names({"vertex_indices", "vertex_index"});
        std::vector<std::vector<FaceIndexT>> f;
        for(auto& name : names) {
            try {
                f = data.getElement("face").getListPropertyAnySign<FaceIndexT>(name);
                break;
            }
            catch(const std::runtime_error& e) {
                // wrMesh<PointT,FaceT,VectorT>ong face index name, trying another
            }
        }

        // This wastes memory but has a greater flexibility
        std::vector<Face> faces(f.size());
        for(int i = 0; i < f.size(); i++) {
            faces[i].x = f[i][0];
            faces[i].y = f[i][1];
            faces[i].z = f[i][2];
        }
        res->faces() = faces;
    }
    
    return res;
}

template <typename P, typename F, typename N, typename U, template<typename> class V>
template <typename PointScalarT, typename FaceIndexT>
void Mesh<P,F,N,U,V>::export_ply(const std::string& path, bool ascii) const
{
    happly::PLYData data;
    
    if(points_.size() == 0) return;

    data.addElement("vertex", points_.size());
    auto& vElement = data.getElement("vertex");
    
    // have to do a copy because of the way happly is implemented
    std::vector<PointScalarT> x(points_.size());
    std::vector<PointScalarT> y(points_.size());
    std::vector<PointScalarT> z(points_.size());
    for(int i = 0; i < points_.size(); i++) {
        x[i] = points_[i].x;
        y[i] = points_[i].y;
        z[i] = points_[i].z;
    }
    vElement.addProperty("x", x);
    vElement.addProperty("y", y);
    vElement.addProperty("z", z);
    
    if(faces_.size() > 0) {
        data.addElement("face", faces_.size());
        // ugly. See alternatives to happly.
        std::vector<std::vector<FaceIndexT>> faces(faces_.size());
        for(size_t i = 0; i < faces.size(); i++) {
            faces[i].resize(3);
            faces[i][0] = faces_[i].x;
            faces[i][1] = faces_[i].y;
            faces[i][2] = faces_[i].z;
        }
        data.getElement("face").addListProperty("vertex_indices", faces);
    }
    
    if(ascii)
        data.write(path, happly::DataFormat::ASCII);
    else
        data.write(path, happly::DataFormat::Binary);
}
#endif //RTAC_CUDACC

}; //namespace rtac

template <typename P, typename F, typename N, typename U, template<typename> class V>
std::ostream& operator<<(std::ostream& os, const rtac::Mesh<P,F,N,U,V>& mesh)
                         
{
    const char* prefix = "\n- ";

    os << "Mesh :"
       << prefix <<  mesh.points().size()  << " points"
       << prefix <<  mesh.faces().size()   << " faces"
       << prefix <<  mesh.normals().size() << " normals"
       << prefix <<  mesh.uvs().size()     << " uvs";

    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_MESH_H_




