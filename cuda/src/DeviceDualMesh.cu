#include <rtac_base/cuda/DeviceDualMesh.h>

namespace rtac {

DualMesh<cuda::DeviceMesh<>>::DualMesh(const cuda::DeviceMesh<>& mesh, unsigned int subdivide) :
    faces_(pow(4,subdivide)*mesh.faces().size())
{
    HostVector<Face> meshFaces(mesh.faces());
    HostVector<Face> dualFaces(mesh.faces().size());
    EdgeSet edges;
    for(unsigned int i = 0; i < meshFaces.size(); i++)
    {
        auto f = meshFaces[i];
        dualFaces[i].x = edges.insert(Edge(f.x, f.y));
        dualFaces[i].y = edges.insert(Edge(f.y, f.z));
        dualFaces[i].z = edges.insert(Edge(f.z, f.x));
    }

    this->reserve(mesh.points().size(),
                  mesh.faces().size(),
                  edges.size(),
                  subdivide);
    cudaMemcpy(points_.data(), mesh.points().data(), 
               sizeof(Point)*mesh.points().size(),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(edges_.data(), edges.data(),
               sizeof(Edge)*edges.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(faces_.data(), dualFaces.data(),
               sizeof(Face)*dualFaces.size(),
               cudaMemcpyHostToDevice);

    this->subdivide(mesh.points().size(), mesh.faces().size(), edges.size(), subdivide);
}

void DualMesh<cuda::DeviceMesh<>>::reserve(unsigned int pointCount, 
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

__global__ void split_edges(unsigned int edgeCount, unsigned int pointCount,
                            Edge* edges, float3* points)
{
    auto i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < edgeCount) {
        Edge e = edges[i];
        auto p0 = points[e.first];
        auto p1 = points[e.second];
        points[pointCount + i] = float3{0.5f*(p0.x + p1.x),
                                         0.5f*(p0.y + p1.y),
                                         0.5f*(p0.z + p1.z)};
        edges[i]             = Edge(e.first,  pointCount + i, i);
        edges[edgeCount + i] = Edge(pointCount + i, e.second, edgeCount + i);
    }
}

__global__ void make_dual_faces(unsigned int faceCount, unsigned int edgeCount,
                                uint3* faces, Edge* edges)
{
    auto i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < faceCount) {
        uint3 f = faces[i];

        // indexes of all new edges that subdivide this face
        auto e0 = edges[f.x];
        auto e1 = edges[f.x + edgeCount];
        auto e2 = edges[f.y];
        auto e3 = edges[f.y + edgeCount];
        auto e4 = edges[f.z];
        auto e5 = edges[f.z + edgeCount];
        auto e6 = Edge(e0.second, e2.second, 2*edgeCount + 3*i);
        auto e7 = Edge(e2.second, e4.second, 2*edgeCount + 3*i + 1);
        auto e8 = Edge(e4.second, e0.second, 2*edgeCount + 3*i + 2);

        edges[e6.index] = e6;
        edges[e7.index] = e7;
        edges[e8.index] = e8;
        f.x = e6.index; f.y = e7.index; f.z = e8.index;
        faces[i] = f;

        auto make_face = [](const Edge& e,
                            const Edge& e0, const Edge& e1,
                            const Edge& e2, const Edge& e3)
        {
            if(e0.is_adjacent(e2)) return uint3{e.index, e0.index, e2.index};
            if(e0.is_adjacent(e3)) return uint3{e.index, e0.index, e3.index};
            if(e1.is_adjacent(e2)) return uint3{e.index, e1.index, e2.index};
            if(e1.is_adjacent(e3)) return uint3{e.index, e1.index, e3.index};
            return uint3{0,0,0};
        };

        unsigned int idx = faceCount + 3*i;
        faces[idx]     = make_face(e6, e0, e1, e2, e3);
        faces[idx + 1] = make_face(e7, e2, e3, e4, e5);
        faces[idx + 2] = make_face(e8, e4, e5, e0, e1);
    }
}


void DualMesh<cuda::DeviceMesh<>>::subdivide(unsigned int pointCount, unsigned int faceCount,
                                             unsigned int edgeCount, unsigned int level)
{
    if(level == 0) {
        return;
    }
    split_edges<<<edgeCount / 256 + 1, 256>>>(edgeCount, pointCount,
                                              edges_.data(), points_.data());
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    make_dual_faces<<<faceCount / 256 + 1, 256>>>(faceCount, edgeCount,
                                                  faces_.data(), edges_.data());
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    subdivide(pointCount + edgeCount, 4*faceCount, 2*edgeCount + 3*faceCount, level - 1);
}

__device__ uint3 make_face(const Edge& e0, const Edge& e1, const Edge&)
{
    // these checks keep triangle oriented the same way
    if(e0.second == e1.first) {
        return uint3{e0.first, e0.second, e1.second};
    }
    else if(e0.first == e1.first) {
        return uint3{e0.second, e0.first, e1.second};
    }
    else if(e0.second == e1.second) {
        return uint3{e0.first, e0.second, e1.first};
    }
    else {
        return uint3{e0.second, e0.first, e1.first};
    }
}

__global__ void make_faces(VectorView<uint3> faces,
                           VectorView<const uint3> dualFaces,
                           VectorView<const Edge> edges)
{
    auto i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < faces.size()) {
        faces[i] = make_face(edges[dualFaces[i].x],
                             edges[dualFaces[i].y],
                             edges[dualFaces[i].z]);
    }
}

cuda::DeviceMesh<>::Ptr DualMesh<cuda::DeviceMesh<>>::create_mesh()
{
    auto mesh = cuda::DeviceMesh<>::Create();
    cuda::DeviceVector<uint3> faces(faces_.size());

    make_faces<<<faces.size() / 256 + 1, 256>>>(faces.view(),
                                                faces_.view(),
                                                edges_.view());
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();

    mesh->points() = points_;
    mesh->faces()  = faces;
    return mesh;
}



} //namespace rtac
