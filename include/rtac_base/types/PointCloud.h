#ifndef _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
#define _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#ifdef RTAC_BASE_PLY_FILES
#include <happly/happly.h>
#endif

#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>
#include <rtac_base/types/PointCloudBase.h>

namespace rtac { namespace types {

// Rtac interface for point cloud type.  Was made to contain a
// pcl::PointCloud but with a soft dependency to pcl.
template <typename PointCloudT = PointCloudBase<Point3D>>
class PointCloud
{
    public:
    
    using PointCloudType = PointCloudT;
    using Ptr            = typename PointCloudT::Ptr;
    using ConstPtr       = typename PointCloudT::ConstPtr;
    using PointType      = typename PointCloudT::PointType;
    using iterator       = typename PointCloudT::VectorType::iterator;
    using const_iterator = typename PointCloudT::VectorType::const_iterator;
    using Pose           = rtac::types::Pose<float>;

    protected:
    
    Ptr pointCloud_;

    public:

    PointCloud();
    PointCloud(const Ptr& pc);
    PointCloud(uint32_t width, uint32_t height = 1);
    PointCloud<PointCloudT> copy() const;

    void resize(size_t n);
    void resize(uint32_t width, uint32_t height);
    void push_back(const PointType& p);
    
    // Implicit conversion to underlying type
    // (allows for direct use in pcl functions)
    operator const PointCloudT&() const;
    operator       PointCloudT&();
    operator ConstPtr() const;
    operator      Ptr();
    const PointCloudT& point_cloud() const;
    PointCloudT&       point_cloud();

    const PointType& at(int col, int row) const;
          PointType& at(int col, int row);
    const PointType& operator()(int col, int row) const;
          PointType& operator()(int col, int row);
    const PointType& at(size_t n) const;
          PointType& at(size_t n);
    const PointType& operator[](size_t n) const;
          PointType& operator[](size_t n);
    const_iterator begin() const;
          iterator begin();
    const_iterator end() const;
          iterator end();
    
    Pose pose() const;
    void set_pose(const Pose& pose);

    size_t size()   const;
    size_t width()  const;
    size_t height() const;
    bool   empty()  const;

    // .ply files
#ifdef RTAC_BASE_PLY_FILES
    static PointCloud<PointCloudT> from_ply(const std::string& path);
    static PointCloud<PointCloudT> from_ply(std::istream& is);
    void export_ply(const std::string& path, bool ascii=false) const;
    void export_ply(std::ostream& os, bool ascii=false) const;
#endif
};

//implementation
template <typename PointCloudT>
PointCloud<PointCloudT>::PointCloud() :
    pointCloud_(NULL)
{}

template <typename PointCloudT>
PointCloud<PointCloudT>::PointCloud(const Ptr& pc) :
    pointCloud_(pc)
{}

template <typename PointCloudT>
PointCloud<PointCloudT>::PointCloud(uint32_t width, uint32_t height) :
    pointCloud_(new PointCloudT(width, height))
{}

template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::copy() const
{
    return PointCloud<PointCloudT>(Ptr(new PointCloudT(*pointCloud_)));
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::resize(size_t n)
{
    if(!pointCloud_)
        pointCloud_ = Ptr(new PointCloudT(n, 1));
    else 
        pointCloud_->resize(n);
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::resize(uint32_t width, uint32_t height)
{
    if(!pointCloud_) {
        pointCloud_ = Ptr(new PointCloudT(width, height));
    }
    else {
        this->resize(width * height);
        pointCloud_->width  = width;
        pointCloud_->height = height;
    }
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::push_back(const PointType& p)
{
    pointCloud_->push_back(p);
}

template <typename PointCloudT>
PointCloud<PointCloudT>::operator const PointCloudT&() const
{
    return *pointCloud_;
}

template <typename PointCloudT>
PointCloud<PointCloudT>::operator PointCloudT&()
{
    return *pointCloud_;
}

template <typename PointCloudT>
PointCloud<PointCloudT>::operator PointCloud<PointCloudT>::ConstPtr() const
{
    return pointCloud_;
}

template <typename PointCloudT>
PointCloud<PointCloudT>::operator PointCloud<PointCloudT>::Ptr()
{
    return pointCloud_;
}

template <typename PointCloudT>
const PointCloudT& PointCloud<PointCloudT>::point_cloud() const
{
    return *pointCloud_;
}

template <typename PointCloudT>
PointCloudT& PointCloud<PointCloudT>::point_cloud()
{
    return *pointCloud_;
}

template <typename PointCloudT>
const typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::at(int col, int row) const
{
    pointCloud_->at(col, row);
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::at(int col, int row)
{
    pointCloud_->at(col, row);
}

template <typename PointCloudT>
const typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::operator()(int col, int row) const
{
    return (*pointCloud_)(col, row);
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::operator()(int col, int row)
{
    return (*pointCloud_)(col, row);
}

template <typename PointCloudT>
const typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::at(size_t n) const
{
    return pointCloud_->at(n);
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::at(size_t n)
{
    return pointCloud_->at(n);
}

template <typename PointCloudT>
const typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::operator[](size_t n) const
{
    return (*pointCloud_)[n];
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::PointType& PointCloud<PointCloudT>::operator[](size_t n)
{
    return (*pointCloud_)[n];
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::const_iterator PointCloud<PointCloudT>::begin() const
{
    return pointCloud_->begin();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::iterator PointCloud<PointCloudT>::begin()
{
    return pointCloud_->begin();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::const_iterator PointCloud<PointCloudT>::end() const
{
    return pointCloud_->end();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::iterator PointCloud<PointCloudT>::end()
{
    return pointCloud_->end();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::Pose PointCloud<PointCloudT>::pose() const
{
    using namespace rtac::types::indexing;
    return Pose(pointCloud_->sensor_origin_(seqN(0,3)),
                pointCloud_->sensor_orientation_);
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::set_pose(const Pose& pose)
{
    pointCloud_->sensor_origin_ << pose.translation(), 1.0f;
    pointCloud_->sensor_orientation_ = pose.orientation();
}

template <typename PointCloudT>
size_t PointCloud<PointCloudT>::size()  const
{
    return pointCloud_->size();
}

template <typename PointCloudT>
size_t PointCloud<PointCloudT>::width()  const
{
    return pointCloud_->width;
}

template <typename PointCloudT>
size_t PointCloud<PointCloudT>::height()  const
{
    return pointCloud_->height;
}

template <typename PointCloudT>
bool PointCloud<PointCloudT>::empty() const
{
    return pointCloud_->empty();
}

// .ply files
#ifdef RTAC_BASE_PLY_FILES
template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::from_ply(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::in);
    if(!f.is_open()) {
        throw std::runtime_error(
            "PointCloud::from_ply : could not open file for reading " + path);
    }
    return PointCloud<PointCloudT>::from_ply(f);
}

template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::from_ply(std::istream& is)
{
    PointCloud<PointCloudT> res;
    
    happly::PLYData data(is);

    // getting point cloud shape.
    uint32_t w = data.getElement("shape").getProperty<uint32_t>("w")[0];
    uint32_t h = data.getElement("shape").getProperty<uint32_t>("h")[0];
    
    // getting pose
    Pose pose;
    pose.translation()(0)  = data.getElement("pose").getProperty<float>("x")[0];
    pose.translation()(1)  = data.getElement("pose").getProperty<float>("y")[0];
    pose.translation()(2)  = data.getElement("pose").getProperty<float>("z")[0];
    pose.orientation().w() = data.getElement("pose").getProperty<float>("qw")[0];
    pose.orientation().x() = data.getElement("pose").getProperty<float>("qx")[0];
    pose.orientation().y() = data.getElement("pose").getProperty<float>("qy")[0];
    pose.orientation().z() = data.getElement("pose").getProperty<float>("qz")[0];

    res = PointCloud<PointCloudT>(w,h);
    res.set_pose(pose);
    
    //loading data (fixed type for now, pcl seems to only use float32)
    auto x = data.getElement("vertex").getProperty<float>("x");
    auto y = data.getElement("vertex").getProperty<float>("y");
    auto z = data.getElement("vertex").getProperty<float>("z");
    
    if(x.size() != res.size()) {
        throw std::runtime_error(
            "PointCloud::from_ply : inconsistent shape and number of vertices");
    }
    
    int i = 0;
    for(auto& p : res) {
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        i++;
    }

    return res;
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::export_ply(const std::string& path, bool ascii) const
{
    std::ofstream f(path, std::ios::binary | std::ios::out);
    if(!f.is_open()) {
        throw std::runtime_error(
            "PointCloud::from_ply : could not open file for writing " + path);
    }
    PointCloud<PointCloudT>::export_ply(f, ascii);
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::export_ply(std::ostream& os, bool ascii) const
{
    if(this->size() <= 0)
        return;

    happly::PLYData data;
    
    // vertices
    data.addElement("vertex", this->size());
    auto& vertices = data.getElement("vertex");
    std::vector<float> x(this->size());
    std::vector<float> y(this->size());
    std::vector<float> z(this->size());
    int i = 0;
    for(auto p : *this) {
        x[i] = p.x;
        y[i] = p.y;
        z[i] = p.z;
        i++;
    }
    vertices.addProperty("x", x);
    vertices.addProperty("y", y);
    vertices.addProperty("z", z);

    // shape
    data.addElement("shape", 1);
    auto& shape = data.getElement("shape");
    // scalars must be vectors in happly interface (consider changing ply lib...)
    std::vector<uint32_t> w({static_cast<uint32_t>(this->width())});
    std::vector<uint32_t> h({static_cast<uint32_t>(this->height())});
    shape.addProperty("w", w);
    shape.addProperty("h", h);
    
    // pose
    data.addElement("pose", 1);
    auto& pose = data.getElement("pose");
    // scalars must be vectors in happly interface (consider changing ply lib...)
    auto p = this->pose();
    std::vector<float> px({p.translation()(0)});
    std::vector<float> py({p.translation()(1)});
    std::vector<float> pz({p.translation()(2)});
    std::vector<float> pqw({p.orientation().w()});
    std::vector<float> pqx({p.orientation().x()});
    std::vector<float> pqy({p.orientation().y()});
    std::vector<float> pqz({p.orientation().z()});
    pose.addProperty("x", px);
    pose.addProperty("y", py);
    pose.addProperty("z", pz);
    pose.addProperty("qw", pqw);
    pose.addProperty("qx", pqx);
    pose.addProperty("qy", pqy);
    pose.addProperty("qz", pqz);
    
    //writing to file
    if(ascii)
        data.write(os, happly::DataFormat::ASCII);
    else
        data.write(os, happly::DataFormat::Binary);
}

#endif

}; //namespace types
}; //namespace rtac

template <typename PointCloudT>
std::ostream& operator<<(std::ostream& os, rtac::types::PointCloud<PointCloudT>& pc)
{
    
    auto precision = os.precision();
    os.precision(2);
    os << "PointCloud : (" << pc.height() << "x" << pc.width() << " points)\n";
    if(pc.size() <= 8) {
        for(auto& p : pc) {
            os << p << "\n";
        }
    }
    else {
        for(int i = 0; i < 3; i++) {
            os << pc[i] << "\n";
        }
        os << "...\n";
        for(int i = pc.size() - 2; i < pc.size(); i++) {
            os << pc[i] << "\n";
        }
    }
    os.precision(precision);

    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
