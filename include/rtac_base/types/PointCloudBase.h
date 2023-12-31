#ifndef _DEF_RTAC_BASE_TYPES_POINTCLOUD_BASE_H_
#define _DEF_RTAC_BASE_TYPES_POINTCLOUD_BASE_H_

#include <iostream>
#include <vector>
#include <memory>

#include <rtac_base/types/Point.h>
#include <rtac_base/types/common.h>

namespace rtac {

/**
 * Default PointCloud template parameter. 
 * 
 * PointCloudBase implements the minimal interface to be used as template type
 * in PointCloud. The interface of this type is a subset of the
 * interface of the
 * [PCL::PointCloud](https://pointclouds.org/documentation/index.html) type.
 */
template <typename PointT = Point3<float>>
class PointCloudBase
{
    public:
    
    using PointType      = PointT;
    using VectorType     = std::vector<PointT>;
    using Ptr            = std::shared_ptr<PointCloudBase<PointT>>;
    using ConstPtr       = std::shared_ptr<const PointCloudBase<PointT>>;
    using iterator       = typename VectorType::iterator;
    using const_iterator = typename VectorType::const_iterator;

    public:

    std::vector<PointT> points;
    uint32_t width;
    uint32_t height;                  /**<If height == 1, PointCloud is deemed unorganized */
    Vector4<float>    sensor_origin_; /**< Position in 3D (Homogeneous coordinates x,y,z,w=1).*/
    Quaternion<float> sensor_orientation_;

    public:

    PointCloudBase();
    PointCloudBase(uint32_t width, uint32_t height = 1);
    Ptr makeShared() const;

    void resize(size_t n);
    void push_back(const PointT& p);

    const PointT& at(int col, int row) const;
          PointT& at(int col, int row);
    const PointT& operator()(int col, int row) const;
          PointT& operator()(int col, int row);
    const PointT& at(size_t n) const;
          PointT& at(size_t n);
    const PointT& operator[](size_t n) const;
          PointT& operator[](size_t n);
    const_iterator begin() const;
          iterator begin();
    const_iterator end() const;
          iterator end();

    size_t size()  const;
    bool   empty() const;
};

//implementation
template <typename PointT>
PointCloudBase<PointT>::PointCloudBase() :
    points(0),
    width(0),
    height(1),
    sensor_origin_({0,0,0,0}),
    sensor_orientation_({1,0,0,0})
{}

template <typename PointT>
PointCloudBase<PointT>::PointCloudBase(uint32_t width, uint32_t height) :
    points(width*height),
    width(width),
    height(height),
    sensor_origin_({0,0,0,0}),
    sensor_orientation_({1,0,0,0})
{}

template <typename PointT>
typename PointCloudBase<PointT>::Ptr PointCloudBase<PointT>::makeShared() const
{
    return Ptr(new PointCloudBase<PointT>(*this));
}

/**
 * Reallocates point buffer to contain n elements.
 * 
 * After the operation, the PointCloud will be unorganized (this->width() == n,
 * this->height() == 1).
 * 
 * @param n New number of points.
 */
template <typename PointT>
void PointCloudBase<PointT>::resize(size_t n)
{
    this->points.resize(n);
    this->width  = this->size();
    this->height = 1;
}

/**
 * Insert a new Point at back of point buffer (this->points).
 *
 * After the operation, the PointCloud will be unorganized (this->width() == this->size(),
 * this->height() == 1).
 *
 * @param p A new point.
 */
template <typename PointT>
void PointCloudBase<PointT>::push_back(const PointT& p)
{
    this->points.push_back(p);
    this->width  = this->size();
    this->height = 1;
}

template <typename PointT>
const PointT& PointCloudBase<PointT>::at(int col, int row) const
{
    return this->at(row*width + col);
}

template <typename PointT>
PointT& PointCloudBase<PointT>::at(int col, int row)
{
    return this->at(row*width + col);
}

template <typename PointT>
const PointT& PointCloudBase<PointT>::operator()(int col, int row) const
{
    return this->at(col, row);
}

template <typename PointT>
PointT& PointCloudBase<PointT>::operator()(int col, int row)
{
    return this->at(col, row);
}

template <typename PointT>
const PointT& PointCloudBase<PointT>::at(size_t n) const
{
    return this->points.at(n);
}

template <typename PointT>
PointT& PointCloudBase<PointT>::at(size_t n)
{
    return this->points.at(n);
}

template <typename PointT>
const PointT& PointCloudBase<PointT>::operator[](size_t n) const
{
    return this->points[n];
}

template <typename PointT>
PointT& PointCloudBase<PointT>::operator[](size_t n)
{
    return this->points[n];
}

/**
 * begin iterator on this->points.
 */
template <typename PointT>
typename PointCloudBase<PointT>::const_iterator PointCloudBase<PointT>::begin() const
{
    return this->points.begin();
}

/**
 * begin iterator on this->points.
 */
template <typename PointT>
typename PointCloudBase<PointT>::iterator PointCloudBase<PointT>::begin()
{
    return this->points.begin();
}

/**
 * end iterator on this->points.
 */
template <typename PointT>
typename PointCloudBase<PointT>::const_iterator PointCloudBase<PointT>::end() const
{
    return this->points.end();
}

/**
 * end iterator on this->points.
 */
template <typename PointT>
typename PointCloudBase<PointT>::iterator PointCloudBase<PointT>::end()
{
    return this->points.end();
}

template <typename PointT>
size_t PointCloudBase<PointT>::size()  const
{
    return this->points.size();
}

/**
 * Checks if PointCloudBase is empty
 *
 * @return Boolean true if is empty.
 */
template <typename PointT>
bool PointCloudBase<PointT>::empty() const
{
    return this->size() == 0;
}

}; //namespace rtac

template <typename PointT>
std::ostream& operator<<(std::ostream& os, const rtac::PointCloudBase<PointT>& pc)
{
    auto precision = os.precision();
    os.precision(2);
    os << "PointCloud : (" << pc.height << "x" << pc.width << " points)\n";
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

#endif //_DEF_RTAC_BASE_TYPES_POINTCLOUD_BASE_H_
