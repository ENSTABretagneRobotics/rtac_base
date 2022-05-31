#ifndef _DEF_RTAC_BASE_EXTERNAL_OBJ_LOADER_H_
#define _DEF_RTAC_BASE_EXTERNAL_OBJ_LOADER_H_

#include <iostream>
#include <memory>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

#include <rtac_base/files.h>
#include <rtac_base/types/Point.h>

//#include <rtac_display/GLMesh.h>

namespace rtac { namespace external {

template <typename T>
class Chunk
{
    public:

    using value_type = T;
    using const_iterator = typename std::vector<T>::const_iterator;

    protected:

    std::vector<T> data_;
    unsigned int currentSize_;

    public:

    Chunk(unsigned int capacity = 100) : 
        data_(capacity),
        currentSize_(0)
    {}

    unsigned int capacity() const { return data_.size(); }
    unsigned int size()     const { return currentSize_; }

    void push_back(const T& value) {
        if(currentSize_ >= this->capacity()) {
            throw std::runtime_error("Chunk is full !");
        }
        data_[currentSize_] = value;
        currentSize_++;
    }

    T operator[](unsigned int idx) const { return data_[idx]; }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const   { return data_.begin() + this->size(); }
};

template <typename T>
class ChunkContainer
{
    public:
    
    using value_type = T;

    class ConstIterator
    {
        protected:

        typename std::list<Chunk<T>>::const_iterator listIt_;
        typename std::list<Chunk<T>>::const_iterator listLast_;
        typename Chunk<T>::const_iterator            chunkIt_;
        typename Chunk<T>::const_iterator            chunkEnd_;
        
        public:

        ConstIterator(typename std::list<Chunk<T>>::const_iterator listIt,
                      typename std::list<Chunk<T>>::const_iterator listLast) :
            listIt_(listIt), listLast_(listLast),
            chunkIt_(listIt->begin()), chunkEnd_(listIt->end())
        {}

        ConstIterator(typename std::list<Chunk<T>>::const_iterator listIt,
                      typename std::list<Chunk<T>>::const_iterator listLast,
                      typename Chunk<T>::const_iterator            chunkIt,
                      typename Chunk<T>::const_iterator            chunkEnd) :
            listIt_(listIt), listLast_(listLast), chunkIt_(chunkIt), chunkEnd_(chunkEnd)
        {}

        ConstIterator& operator++() {

            if(chunkIt_ + 1 == chunkEnd_) {
                if(listIt_ == listLast_) {
                    chunkIt_++;
                }
                else {
                    listIt_++;
                    chunkIt_  = listIt_->begin();
                    chunkEnd_ = listIt_->end();
                }
            }
            else if(chunkIt_ != chunkEnd_) {
                chunkIt_++;
            }

            return *this;
        }

        bool operator==(const ConstIterator& other) const {
            return listIt_ == other.listIt_ && chunkIt_ == other.chunkIt_;
        }

        bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }

        const T& operator*() const {
            return *chunkIt_;
        }
    };
    
    protected:

    unsigned int chunkSize_;
    std::list<Chunk<T>> chunks_;

    public:

    ChunkContainer(unsigned int chunkSize = 100) :
        chunkSize_(chunkSize),
        chunks_(1, Chunk<T>(chunkSize_))
    {}

    size_t size() const {
        size_t size = 0;
        for(auto chunk : chunks_) {
            size += chunk.size();
        }
        return size;
    }

    void clear() {
        chunks_.clear();
        chunks_.push_back(Chunk<T>(chunkSize_));
    }

    void push_back(T value) {
        if(chunks_.back().size() >= chunkSize_) {
            chunks_.push_back(Chunk<T>(chunkSize_));
        }
        chunks_.back().push_back(value);
    }

    ConstIterator begin() const {
        return ConstIterator(chunks_.begin(), std::prev(chunks_.end()));
    }

    ConstIterator end() const {
        return ConstIterator(std::prev(chunks_.end()), std::prev(chunks_.end()),
                             std::prev(chunks_.end())->end(),
                             std::prev(chunks_.end())->end());
    }

    std::vector<T> to_vector() const
    {
        std::vector<T> out(this->size());
        size_t idx = 0;
        for(auto v : *this) {
            out[idx] = v;
            idx++;
        }

        return out;
    }
};

struct VertexId
{
    uint32_t p;
    uint32_t u;
    uint32_t n;
    mutable uint32_t id;

    bool operator<(const VertexId& other) const {
        if(p < other.p) {
            return true;
        }
        else if(p > other.p) {
            return false;
        }
        else if(u < other.u) {
            return true;
        }
        else if(u > other.u) {
            return false;
        }
        else if(n < other.n) {
            return true;
        }
        else {
            return false;
        }
    }
};

struct MtlMaterial {

    using Color = rtac::types::Point3<float>;
    
    std::string name;
    Color Ka;           // ambient  color
    Color Kd;           // diffuse  color
    Color Ks;           // specular color
    float Ns;           // specular weight
    float d;            // Transparency
    unsigned int illum; // illumination model
    std::string map_Kd; // texture path

    void clear() {
        name   = "";
        Ka     = {0.0f,0.0f,0.0f};
        Kd     = {0.0f,0.0f,0.0f};
        Ks     = {0.0f,0.0f,0.0f};
        Ns     = 0;
        d      = 0;
        illum  = 0;
        map_Kd = "";
    }
};

class ObjLoader
{
    public:

    using Point  = rtac::types::Point3<float>;
    using Face   = rtac::types::Point3<uint32_t>;
    using UV     = rtac::types::Point2<float>;
    using Normal = rtac::types::Point3<float>;

    protected:

    std::string datasetPath_;
    std::string objPath_;
    std::string mtlPath_;

    std::vector<Point>  points_;
    std::vector<UV>     uvs_;
    std::vector<Normal> normals_;
    std::set<VertexId>  vertices_;
    std::vector<std::string> groupNames_;
    std::map<std::string,std::vector<Face>> faceGroups_;

    std::map<std::string,MtlMaterial> materials_;

    public:

    ObjLoader(const std::string& datasetPath);

    void load_geometry(unsigned int chunkSize = 100);
    void parse_mtl();

    std::string dataset_path() const { return datasetPath_; }
    std::string obj_path()     const { return objPath_; }
    std::string mtl_path()     const { return mtlPath_; }

    const std::vector<Point>&  points()   const { return points_; }
    const std::vector<UV>&     uvs()      const { return uvs_; }
    const std::vector<Normal>& normals()  const { return normals_; }
    const std::set<VertexId>&  vertices() const { return vertices_; }
    const std::map<std::string,std::vector<Face>>& faces() const { return faceGroups_; }
    const std::map<std::string,MtlMaterial>& materials() const { return materials_; }
    const MtlMaterial& material(const std::string& name) const {
        return materials_.at(name);
    }
    

    template <class MeshT>
    std::map<std::string, typename MeshT::Ptr> create_meshes();
    template <class MeshT>
    typename MeshT::Ptr create_single_mesh();
};

template <class MeshT>
std::map<std::string, typename MeshT::Ptr> ObjLoader::create_meshes()
{
    std::map<std::string, typename MeshT::Ptr> meshes;

    for(auto name : groupNames_) {

        auto mesh = MeshT::Create();

        {
            std::vector<typename MeshT::Point> points(vertices_.size());
            for(auto v : vertices_) {
                points[v.id].x = points_[v.p].x;
                points[v.id].y = points_[v.p].y;
                points[v.id].z = points_[v.p].z;
            }
            mesh->points() = points;
        }

        if(uvs_.size() > 0) {
            std::vector<typename MeshT::UV> uvs(vertices_.size());
            for(auto v : vertices_) {
                uvs[v.id].x = uvs_[v.u].x;
                uvs[v.id].y = uvs_[v.u].y;
            }
            mesh->uvs() = uvs;
        }

        if(normals_.size() > 0) {
            std::vector<typename MeshT::Normal> normals(vertices_.size());
            for(auto v : vertices_) {
                normals[v.id].x = normals_[v.n].x;
                normals[v.id].y = normals_[v.n].y;
                normals[v.id].z = normals_[v.n].z;
            }
            mesh->normals() = normals;
        }

        if(faceGroups_[name].size() > 0) {
            auto data = &faceGroups_[name];
            std::vector<typename MeshT::Face> faces(data->size());
            for(int i = 0; i < faces.size(); i++) {
                faces[i].x = (*data)[i].x;
                faces[i].y = (*data)[i].y;
                faces[i].z = (*data)[i].z;
            }
            mesh->faces() = faces;
        }

        meshes[name] = mesh;
    }
    return meshes;
}

template <class MeshT>
typename MeshT::Ptr ObjLoader::create_single_mesh()
{
    auto mesh = MeshT::Create();


    {
        std::vector<typename MeshT::Point> points(vertices_.size());
        for(auto v : vertices_) {
            points[v.id].x = points_[v.p].x;
            points[v.id].y = points_[v.p].y;
            points[v.id].z = points_[v.p].z;
        }
        mesh->points() = points;
    }

    if(normals_.size() > 0) {
        std::vector<typename MeshT::Normal> normals(vertices_.size());
        for(auto v : vertices_) {
            normals[v.id].x = normals_[v.n].x;
            normals[v.id].y = normals_[v.n].y;
            normals[v.id].z = normals_[v.n].z;
        }
        mesh->normals() = normals;
    }

    if(faceGroups_.size() > 0) {
        std::vector<typename MeshT::Face> faces;
        for(auto name : groupNames_) {
            auto data = &faceGroups_[name];
            for(int i = 0; i < data->size(); i++) {
                typename MeshT::Face f;
                f.x = (*data)[i].x;
                f.y = (*data)[i].y;
                f.z = (*data)[i].z;
                faces.push_back(f);
            }
        }
        mesh->faces() = faces;
    }
    return mesh;
}

}; //namespace display
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os, const rtac::external::MtlMaterial& mat)
{
    os << "newmtl "  << mat.name
       << "\nKa " << mat.Ka.x << " " << mat.Ka.x << " " << mat.Ka.x
       << "\nKd " << mat.Kd.y << " " << mat.Kd.y << " " << mat.Kd.y
       << "\nKs " << mat.Ks.z << " " << mat.Ks.z << " " << mat.Ks.z
       << "\nNs "    << mat.Ns
       << "\nd  "    << mat.d
       << "\nillum " << mat.illum
       << "\nmap_Kd " << mat.map_Kd;

    return os;
}

#endif //_DEF_RTAC_BASE_EXTERNAL_OBJ_LOADER_H_
