#ifndef _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
#define _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_

#include <iostream>
#include <vector>
#include <memory>

#include <rtac_base/types/common.h>
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

    protected:
    
    Ptr pointCloud_;

    public:

    PointCloud();
    PointCloud(const Ptr& pc);
    PointCloud(uint32_t width, uint32_t height = 1);

    void resize(size_t n);
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
    
    size_t size()   const;
    size_t width()  const;
    size_t height() const;
    bool   empty()  const;
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
void PointCloud<PointCloudT>::resize(size_t n)
{
    pointCloud_->resize(n);
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
