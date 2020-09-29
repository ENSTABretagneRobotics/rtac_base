#ifndef _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
#define _DEF_RTAC_BASE_TYPES_POINTCLOUD_H_

#include <iostream>

#include <rtac_base/types/common.h>

namespace rtac { namespace types {

using namespace rtac::types::indexing;

// Tp stands for pointType (usually float)
// D is the dimension of points space (usually 3
template <typename Tp = float, size_t D = 3>
class PointCloud
{
    protected:

    Array<Tp,D> points_;

    public:

    PointCloud();
    PointCloud(size_t numPoints);
    PointCloud(const Array<Tp,D>& points);

    Map<Matrix<Tp>> points();

    Map<const Matrix<Tp>> points() const;
    size_t num_points() const;
};

//implementation
template <typename Tp, size_t D>
PointCloud<Tp,D>::PointCloud() :
    points_(0)
{}

template <typename Tp, size_t D>
PointCloud<Tp,D>::PointCloud(size_t numPoints) :
    points_(numPoints)
{}

template <typename Tp, size_t D>
PointCloud<Tp,D>::PointCloud(const Array<Tp,D>& points) :
    points_(points)
{}

template <typename Tp, size_t D>
Map<Matrix<Tp>> PointCloud<Tp,D>::points()
{
    if (this->num_points() <= 0)
        return Map<Matrix<Tp>>(NULL, 0, D);
    return Map<Matrix<Tp>>(points_.data(), this->num_points(), D);
}

template <typename Tp, size_t D>
Map<const Matrix<Tp>> PointCloud<Tp,D>::points() const
{
    if (this->num_points() <= 0)
        return Map<const Matrix<Tp>>(NULL, 0, D);
    return Map<const Matrix<Tp>>(points_.data(), this->num_points(), D);
}

template <typename Tp, size_t D>
size_t PointCloud<Tp,D>::num_points() const
{
    return points_.rows();
}

}; //namespace types
}; //namespace rtac

template <typename Tp, size_t D>
std::ostream& operator<<(std::ostream& os, rtac::types::PointCloud<Tp,D>& pc)
{
    using namespace rtac::types::indexing;
    os << "PointCloud : (" << pc.num_points() << " points)\n";
    if(pc.num_points() < 8) {
        os << pc.points() << "\n";
    }
    else {
        os << pc.points()(seqN(0,3), all) << "\n...\n"
           << pc.points()(seq(last-2, last), all) << "\n";
    }
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_POINTCLOUD_H_
