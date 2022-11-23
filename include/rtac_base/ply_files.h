#ifndef _DEF_RTAC_BASE_PLY_FILES_H_
#define _DEF_RTAC_BASE_PLY_FILES_H_

#include <iostream>
#include <vector>

#include <rtac_base/happly.h>
#include <rtac_base/types/Pose.h>
#include <rtac_base/types/Shape.h>
#include <rtac_base/types/Rectangle.h>


namespace rtac { namespace ply {

happly::PLYData New();
happly::Element& new_element(happly::PLYData& data, const std::string& name,
                             bool overwrite = false);

void write(std::ostream& os, happly::PLYData& data, bool ascii = false);
void write(const std::string& path, happly::PLYData& data, bool ascii = false);
happly::PLYData read(const std::string& path);
happly::PLYData read(std::istream& is);

// collection of functions to export/import rtac types to/from ply files.
template <typename T>
Pose<T> get_pose(happly::PLYData& data, const::std::string& name = "pose");
template <typename T>
void add_pose(happly::PLYData& data, const Pose<T>& pose, 
              const std::string& name = "pose", bool overwrite = false);

template <typename T>
Shape<T> get_shape(happly::PLYData& data, const::std::string& name = "shape");
template <typename T>
void add_shape(happly::PLYData& data, const Shape<T>& shape, 
               const std::string& name = "shape", bool overwrite = false);

template <typename T>
Rectangle<T> get_rectangle(happly::PLYData& data,
                                  const::std::string& name = "rectangle");
template <typename T>
void add_rectangle(happly::PLYData& data, const Rectangle<T>& rectangle, 
                   const std::string& name = "rectangle", bool overwrite = false);

}; //namespace ply
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, happly::PLYData& data);


// implementation NO DECLARATIONS BEYOND THIS POINT ////////////////////////
namespace rtac { namespace ply {

template <typename T>
Pose<T> get_pose(happly::PLYData& data, const std::string& name)
{
    auto& element = data.getElement(name);
    Pose<T> res;
    res.translation()(0)  = element.getProperty<T>("x")[0];
    res.translation()(1)  = element.getProperty<T>("y")[0];
    res.translation()(2)  = element.getProperty<T>("z")[0];
    typename Pose<T>::Quat q;
    q.w() = element.getProperty<T>("qw")[0];
    q.x() = element.getProperty<T>("qx")[0];
    q.y() = element.getProperty<T>("qy")[0];
    q.z() = element.getProperty<T>("qz")[0];
    res.set_orientation(q);
    return res;
}

template <typename T>
void add_pose(happly::PLYData& data, const Pose<T>& pose, 
              const std::string& name, bool overwrite)
{
    auto& element = new_element(data, name, overwrite);

    // scalars must be vectors in happly interface (consider changing ply lib...)
    std::vector<T> px({pose.translation()(0)});
    std::vector<T> py({pose.translation()(1)});
    std::vector<T> pz({pose.translation()(2)});

    auto q = pose.quaternion();
    std::vector<T> pqw({q.w()});
    std::vector<T> pqx({q.x()});
    std::vector<T> pqy({q.y()});
    std::vector<T> pqz({q.z()});
    element.addProperty("x", px);
    element.addProperty("y", py);
    element.addProperty("z", pz);
    element.addProperty("qw", pqw);
    element.addProperty("qx", pqx);
    element.addProperty("qy", pqy);
    element.addProperty("qz", pqz);
}

template <typename T>
Shape<T> get_shape(happly::PLYData& data, const::std::string& name)
{
    auto& element = data.getElement(name);
    return Shape<T>({element.getProperty<T>("w")[0],
                            element.getProperty<T>("h")[0]});
}

template <typename T>
void add_shape(happly::PLYData& data, const Shape<T>& shape, 
               const std::string& name, bool overwrite)
{
    auto& element = new_element(data, name, overwrite);

    std::vector<T> w({shape.width});
    std::vector<T> h({shape.height});
    element.addProperty("w", w);
    element.addProperty("h", h);
}

template <typename T>
Rectangle<T> get_rectangle(happly::PLYData& data, const::std::string& name)
{
    auto& element = data.getElement(name);
    return Rectangle<T>({element.getProperty<T>("left")[0],
                                element.getProperty<T>("right")[0],
                                element.getProperty<T>("bottom")[0],
                                element.getProperty<T>("top")[0]});
}

template <typename T>
void add_rectangle(happly::PLYData& data, const Rectangle<T>& rectangle, 
                   const std::string& name, bool overwrite)
{
    auto& element = new_element(data, name, overwrite);

    std::vector<T> left(  {rectangle.left});
    std::vector<T> right( {rectangle.right});
    std::vector<T> bottom({rectangle.bottom});
    std::vector<T> top(   {rectangle.top});
    element.addProperty("left",   left);
    element.addProperty("right",  right);
    element.addProperty("bottom", bottom);
    element.addProperty("top",    top);
}

}; //namespace ply
}; //namespace rtac

#endif //_DEF_RTAC_BASE_PLY_FILES_H_
