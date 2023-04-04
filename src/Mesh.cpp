#include <rtac_base/types/Mesh.h>
#include <rtac_base/types/DualMesh.h>

namespace rtac {

Mesh<>::Ptr make_icosphere(unsigned int level, float scale)
{
    DualMesh<Mesh<>> dual(*Mesh<>::icosahedron(), level);
    auto mesh = dual.create_mesh();
    for(auto& p : mesh->points()) {
        auto a = scale / sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        p.x *= a;
        p.y *= a;
        p.z *= a;
    }
    return mesh;
}

}// namespace rtac


