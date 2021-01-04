#ifndef _DEF_RTAC_BASE_PLY_FILES_H_
#define _DEF_RTAC_BASE_PLY_FILES_H_

#include <iostream>
#include <vector>

#include <happly.h>

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
types::Pose<T> get_pose(happly::PLYData& data, const::std::string& name = "pose");
template <typename T>
void add_pose(happly::PLYData& data, const types::Pose<T>& pose, 
              const std::string& name = "pose", bool overwrite = false);

template <typename T>
types::Shape<T> get_shape(happly::PLYData& data, const::std::string& name = "shape");
template <typename T>
void add_shape(happly::PLYData& data, const types::Shape<T>& shape, 
               const std::string& name = "shape", bool overwrite = false);

template <typename T>
types::Rectangle<T> get_rectangle(happly::PLYData& data,
                                  const::std::string& name = "rectangle");
template <typename T>
void add_rectangle(happly::PLYData& data, const types::Rectangle<T>& rectangle, 
                   const std::string& name = "rectangle", bool overwrite = false);

}; //namespace ply
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, happly::PLYData& data);


// implementation NO DECLARATIONS BEYOND THIS POINT ////////////////////////
namespace rtac { namespace ply {

template <typename T>
types::Pose<T> get_pose(happly::PLYData& data, const std::string& name)
{
    auto& element = data.getElement(name);
    types::Pose<T> res;
    res.translation()(0)  = element.getProperty<T>("x")[0];
    res.translation()(1)  = element.getProperty<T>("y")[0];
    res.translation()(2)  = element.getProperty<T>("z")[0];
    res.orientation().w() = element.getProperty<T>("qw")[0];
    res.orientation().x() = element.getProperty<T>("qx")[0];
    res.orientation().y() = element.getProperty<T>("qy")[0];
    res.orientation().z() = element.getProperty<T>("qz")[0];
    return res;
}

template <typename T>
void add_pose(happly::PLYData& data, const types::Pose<T>& pose, 
              const std::string& name, bool overwrite)
{
    auto& element = new_element(data, name, overwrite);

    // scalars must be vectors in happly interface (consider changing ply lib...)
    std::vector<T> px({pose.translation()(0)});
    std::vector<T> py({pose.translation()(1)});
    std::vector<T> pz({pose.translation()(2)});
    std::vector<T> pqw({pose.orientation().w()});
    std::vector<T> pqx({pose.orientation().x()});
    std::vector<T> pqy({pose.orientation().y()});
    std::vector<T> pqz({pose.orientation().z()});
    element.addProperty("x", px);
    element.addProperty("y", py);
    element.addProperty("z", pz);
    element.addProperty("qw", pqw);
    element.addProperty("qx", pqx);
    element.addProperty("qy", pqy);
    element.addProperty("qz", pqz);
}

template <typename T>
types::Shape<T> get_shape(happly::PLYData& data, const::std::string& name)
{
    auto& element = data.getElement(name);
    return types::Shape<T>({element.getProperty<T>("w")[0],
                            element.getProperty<T>("h")[0]});
}

template <typename T>
void add_shape(happly::PLYData& data, const types::Shape<T>& shape, 
               const std::string& name, bool overwrite)
{
    auto& element = new_element(data, name, overwrite);

    std::vector<T> w({shape.width});
    std::vector<T> h({shape.height});
    element.addProperty("w", w);
    element.addProperty("h", h);
}

template <typename T>
types::Rectangle<T> get_rectangle(happly::PLYData& data, const::std::string& name)
{
    auto& element = data.getElement(name);
    return types::Rectangle<T>({element.getProperty<T>("left")[0],
                                element.getProperty<T>("right")[0],
                                element.getProperty<T>("bottom")[0],
                                element.getProperty<T>("top")[0]});
}

template <typename T>
void add_rectangle(happly::PLYData& data, const types::Rectangle<T>& rectangle, 
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
