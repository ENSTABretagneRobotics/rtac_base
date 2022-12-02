#ifndef _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
#define _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include <rtac_base/ply_files.h>

#include <rtac_base/types/Point.h>
#include <rtac_base/types/common.h>
#include <rtac_base/types/Pose.h>
#include <rtac_base/types/PointCloudBase.h>

namespace rtac {

/**
 * Interface template for manipulating pointclouds in the RTAC framework.
 *
 * This template is made to be a soft interface to
 * [PCL](https://pointclouds.org/documentation/index.html). The
 * rtac::PointCloudBase type is provided to enable the use of this
 * template without linking with PCL.
 *
 * @tparam PointCloudT a pointcloud type holding the point data. Compatible
 * with the [PCL::PointCloud](https://pointclouds.org/documentation/index.html)
 * type.
*/
template <typename PointCloudT = PointCloudBase<Point3<float>>>
class PointCloud
{
    public:
    
    using PointCloudType = PointCloudT;
    using Ptr            = typename PointCloudT::Ptr;
    using ConstPtr       = typename PointCloudT::ConstPtr;
    using PointType      = typename PointCloudT::PointType;
    using iterator       = typename PointCloudT::VectorType::iterator;
    using const_iterator = typename PointCloudT::VectorType::const_iterator;
    using Pose           = rtac::Pose<float>;
    using Shape          = rtac::Shape<uint32_t>;

    protected:
    
    Ptr pointCloud_;

    public:

    PointCloud();
    PointCloud(const Ptr& pc);
    PointCloud(uint32_t width, uint32_t height = 1);
    PointCloud(const Shape& shape);
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
    Shape shape() const;

    size_t size()   const;
    size_t width()  const;
    size_t height() const;
    bool   empty()  const;

    // .ply files
    static PointCloud<PointCloudT> from_ply(const std::string& path);
    static PointCloud<PointCloudT> from_ply(std::istream& is);
    static PointCloud<PointCloudT> from_ply(happly::PLYData& data);
    void export_ply(const std::string& path, bool ascii=false) const;
    void export_ply(std::ostream& os, bool ascii=false) const;
    happly::PLYData export_ply() const;
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
PointCloud<PointCloudT>::PointCloud(const Shape& shape) :
    PointCloud(shape.width, shape.height)
{}

template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::copy() const
{
    return PointCloud<PointCloudT>(Ptr(new PointCloudT(*pointCloud_)));
}

/**
 * Reallocates point buffer to contain n elements.
 * 
 * After the operation, the PointCloud will be unorganized (this->width() == n,
 * this->height() == 1).
 * 
 * @param n New number of points.
 */
template <typename PointCloudT>
void PointCloud<PointCloudT>::resize(size_t n)
{
    if(!pointCloud_)
        pointCloud_ = Ptr(new PointCloudT(n, 1));
    else 
        pointCloud_->resize(n);
}

/**
 * Reallocates points data to contain width x height elements.
 * 
 * After the operation, the PointCloud will be organized (this->width() == width,
 * this->height() == height).
 * 
 * @param width  New width of PointCloud
 * @param height New height of PointCloud
 */
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

/**
 * Insert a new Point at back of point buffer.
 *
 * After the operation, the PointCloud will be unorganized (this->width() == this->size(),
 * this->height() == 1).
 *
 * @param p A new point.
 */
template <typename PointCloudT>
void PointCloud<PointCloudT>::push_back(const PointType& p)
{
    pointCloud_->push_back(p);
}

/**
 * Implicit cast to underlying type (const).
 * 
 * If PointCloudT is
 * [PCL::PointCloud](https://pointclouds.org/documentation/index.html), this
 * method allows to use this type in PCL functions without explicit type
 * conversion.
 *
 * @return A const reference to the underlying PointCloudT.
 */
template <typename PointCloudT>
PointCloud<PointCloudT>::operator const PointCloudT&() const
{
    return *pointCloud_;
}

/**
 * Implicit cast to underlying type.
 * 
 * If PointCloudT is
 * [PCL::PointCloud](https://pointclouds.org/documentation/index.html), this
 * method allows to use this type in PCL functions without explicit type
 * conversion.
 *
 * @return A reference to the underlying PointCloudT.
 */
template <typename PointCloudT>
PointCloud<PointCloudT>::operator PointCloudT&()
{
    return *pointCloud_;
}

/**
 * Implicit cast to underlying pointer.
 *
 * @return A pointer to the underlying PointCloudT.
 */
template <typename PointCloudT>
PointCloud<PointCloudT>::operator PointCloud<PointCloudT>::ConstPtr() const
{
    return pointCloud_;
}

/**
 * Implicit cast to underlying pointer.
 *
 * @return A pointer to the underlying PointCloudT.
 */
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

/**
 * begin iterator on points container.
 */
template <typename PointCloudT>
typename PointCloud<PointCloudT>::const_iterator PointCloud<PointCloudT>::begin() const
{
    return pointCloud_->begin();
}

/**
 * begin iterator on points container.
 */
template <typename PointCloudT>
typename PointCloud<PointCloudT>::iterator PointCloud<PointCloudT>::begin()
{
    return pointCloud_->begin();
}

/**
 * end iterator on points container.
 */
template <typename PointCloudT>
typename PointCloud<PointCloudT>::const_iterator PointCloud<PointCloudT>::end() const
{
    return pointCloud_->end();
}

/**
 * end iterator on points container.
 */
template <typename PointCloudT>
typename PointCloud<PointCloudT>::iterator PointCloud<PointCloudT>::end()
{
    return pointCloud_->end();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::Pose PointCloud<PointCloudT>::pose() const
{
    using namespace rtac::indexing;
    return Pose::from_quaternion(pointCloud_->sensor_orientation_,
                                 pointCloud_->sensor_origin_(seqN(0,3)));
}

template <typename PointCloudT>
void PointCloud<PointCloudT>::set_pose(const Pose& pose)
{
    pointCloud_->sensor_origin_ << pose.translation(), 1.0f;
    pointCloud_->sensor_orientation_ = pose.orientation();
}

template <typename PointCloudT>
typename PointCloud<PointCloudT>::Shape PointCloud<PointCloudT>::shape() const
{
    return Shape({static_cast<uint32_t>(this->width()),
                  static_cast<uint32_t>(this->height())});
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

/**
 * Checks if PointCloud is empty
 *
 * @return Boolean true if is empty.
 */
template <typename PointCloudT>
bool PointCloud<PointCloudT>::empty() const
{
    return pointCloud_->empty();
}

/**
 * Loads a PointCloud from a .ply file.
 * 
 * The ply file must contain "shape", "pose" and "vertex" elements.
 * 
 * @param path Path to the .ply file to load from.
 *
 * @return A newly allocated PointCloud.
 */
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

/**
 * Loads a PointCloud from a .ply file.
 * 
 * The ply file must contain "shape", "pose" and "vertex" elements.
 * 
 * @param is File descriptor to load from.
 *
 * @return A newly allocated PointCloud.
 */
template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::from_ply(std::istream& is)
{
    happly::PLYData data(is);
    return from_ply(data);
}

/**
 * Loads a PointCloud from a .ply file.
 * 
 * The ply file must contain "shape", "pose" and "vertex" elements.
 * 
 * @param data Data to load from.
 *
 * @return A newly allocated PointCloud.
 */
template <typename PointCloudT>
PointCloud<PointCloudT> PointCloud<PointCloudT>::from_ply(happly::PLYData& data)
{
    PointCloud<PointCloudT> res;
    
    // getting point cloud shape.
    auto shape = ply::get_shape<uint32_t>(data);

    res = PointCloud<PointCloudT>(shape);
    res.set_pose(ply::get_pose<float>(data));
    
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

/**
 * Export PointCloud to a .ply file.
 *
 * This will populate the "pose", "shape" and "vertex" elements in the .ply
 * file.
 *
 * @param path Path of exported .ply file.
 * @param ascii Export in ascii (true) or binary (false) format.
 */
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

/**
 * Export PointCloud to a .ply file.
 *
 * This will populate the "pose", "shape" and "vertex" elements in the .ply
 * file.
 *
 * @param os File descriptor to write to.
 * @param ascii Export in ascii (true) or binary (false) format.
 */
template <typename PointCloudT>
void PointCloud<PointCloudT>::export_ply(std::ostream& os, bool ascii) const
{
    if(this->size() <= 0)
        return;
    
    auto data = this->export_ply();

    //writing to file
    if(ascii)
        data.write(os, happly::DataFormat::ASCII);
    else
        data.write(os, happly::DataFormat::Binary);
}

/**
 * Export PointCloud to a happly::PLYData data structure.
 *
 * This will populate the "pose", "shape" and "vertex" elements in the .ply
 * structure.
 *
 * @param ascii Export in ascii (true) or binary (false) format.
 *
 * @return A new happly::PLYData structure containing data from the PointCloud.
 */
template <typename PointCloudT>
happly::PLYData PointCloud<PointCloudT>::export_ply() const
{
    happly::PLYData data;

    if(this->size() <= 0)
        return data;

    ply::add_shape(data, this->shape());
    ply::add_pose(data, this->pose());
    
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

    return data;
}

}; //namespace rtac

template <typename PointCloudT>
std::ostream& operator<<(std::ostream& os, rtac::PointCloud<PointCloudT>& pc)
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
