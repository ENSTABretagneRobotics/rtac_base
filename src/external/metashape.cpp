#include <rtac_base/external/metashape.h>

#include <sstream>
#include <string>

#include <rtac_base/external/tinyxml2.h>

namespace rtac { namespace external {

using namespace tinyxml2;

MetashapeOutput::PoseInfo parse_pose(XMLElement* cam)
{
    if(!cam) {
        throw std::runtime_error("camera element in nullptr");
    }

    MetashapeOutput::PoseInfo info;

    if(XML_SUCCESS != cam->QueryIntAttribute("id", &info.id)) {
        throw std::runtime_error("No id attribute in camera element");
    }
    if(XML_SUCCESS != cam->QueryIntAttribute("sensor_id", &info.sensorId)) {
        throw std::runtime_error("No sensorId attribute in camera element");
    }
    if(XML_SUCCESS != cam->QueryIntAttribute("component_id", &info.componentId)) {
        throw std::runtime_error("No sensorId attribute in camera element");
    }
    
    auto label = cam->Attribute("label");
    if(!label) {
        throw std::runtime_error("No label attribute in camera element");
    }
    info.label = label;

    auto transform = cam->FirstChildElement("transform");
    if(!transform) {
        std::ostringstream oss;
        oss << "XML : no transform element in camera element";
        throw std::runtime_error(oss.str());
    }

    auto transformData = transform->GetText();
    if(!transformData) {
        std::ostringstream oss;
        oss << "XML : no text in transform element";
        throw std::runtime_error(oss.str());
    }

    Pose<float>::Mat4 mat;
    std::string token, transformStr(transformData);
    std::istringstream iss(transformStr);
    for(int r = 0; r < 4; r++) {
        for(int c = 0; c < 4; c++) {
            if(!iss) {
                std::ostringstream oss;
                oss << "Invalid transform format : '" << transformData << "'";
                throw std::runtime_error(oss.str());
            }
            std::getline(iss, token, ' ');
            mat(r,c) = std::stof(token);
        }
    }

    info.pose = Pose<float>::from_homogeneous_matrix(mat);

    return info;
}

MetashapeOutput::Component parse_component(XMLElement* element)
{
    MetashapeOutput::Component component;

    if(XML_SUCCESS != element->QueryIntAttribute("id", &component.id)) {
        throw std::runtime_error("No id attribute in component element");
    }
    if(auto str = element->Attribute("label")) {
        component.label = str;
    }

    if(auto tr = element->FirstChildElement("transform")) {
        if(auto rot = tr->FirstChildElement("rotation")) {
            if(auto rotStr = rot->GetText()) {
                std::string data(rotStr);
                std::istringstream iss(data);
                std::string token;
                Pose<float>::Mat3 mat;
                for(int r = 0; r < 3; r++) { for(int c = 0; c < 3; c++) {
                    if(!std::getline(iss, token, ' ')) {
                        throw std::runtime_error("Invalid transform R format");
                    }
                    mat(r,c) = std::stof(token);
                }}
                component.transform.set_orientation(mat);
            }
        }
        if(auto t = tr->FirstChildElement("translation")) {
            if(auto tStr = t->GetText()) {
                std::string data(tStr);
                std::istringstream iss(data);
                std::string token;
                Pose<float>::Vec3 vec;
                for(int r = 0; r < 3; r++) {
                    if(!std::getline(iss, token, ' ')) {
                        throw std::runtime_error("Invalid transform T format");
                    }
                    vec(r) = std::stof(token);
                }
                component.transform.set_translation(vec);
            }
        }
        if(auto scale = tr->FirstChildElement("scale")) {
            if(auto tStr = scale->GetText()) {
                component.scale = std::stof(tStr);
            }
        }
    }

    if(auto reg = element->FirstChildElement("region")) {
        if(auto center = reg->FirstChildElement("center")) {
            if(auto cStr = center->GetText()) {
                std::string data(cStr);
                std::istringstream iss(data);
                std::string token;
                Pose<float>::Vec3 vec;
                for(int r = 0; r < 3; r++) {
                    if(!std::getline(iss, token, ' ')) {
                        throw std::runtime_error("Invalid region center format");
                    }
                    vec(r) = std::stof(token);
                }
                component.regionCenter = vec;
            }
        }
        if(auto size = reg->FirstChildElement("size")) {
            if(auto sStr = size->GetText()) {
                std::string data(sStr);
                std::istringstream iss(data);
                std::string token;
                Pose<float>::Vec3 vec;
                for(int r = 0; r < 3; r++) {
                    if(!std::getline(iss, token, ' ')) {
                        throw std::runtime_error("Invalid region size format");
                    }
                    vec(r) = std::stof(token);
                }
                component.regionSize = vec;
            }
        }
        if(auto rot = reg->FirstChildElement("R")) {
            if(auto rotStr = rot->GetText()) {
                std::string data(rotStr);
                std::istringstream iss(data);
                std::string token;
                Pose<float>::Mat3 mat;
                for(int r = 0; r < 3; r++) { for(int c = 0; c < 3; c++) {
                    if(!std::getline(iss, token, ' ')) {
                        throw std::runtime_error("Invalid region R format");
                    }
                    mat(r,c) = std::stof(token);
                }}
                component.regionRot = mat;
            }
        }
    }

    return component;
}

std::deque<MetashapeOutput::Component> 
    parse_components(XMLElement* chunk)
{
    std::deque<MetashapeOutput::Component> components;

    auto element = chunk->FirstChildElement("components");
    if(!element) {
        std::cerr << "No 'components' element in metashape pose file" << std::endl;
        return components;
    }

    auto compElement = element->FirstChildElement("component");
    while(compElement) {
        components.push_back(parse_component(compElement));
        compElement = compElement->NextSiblingElement("component");
    }

    return components;
}

void MetashapeOutput::load_file(const std::string& filename)
{
    filename_ = filename;
    std::cout << "Parsing " << filename_ << std::endl;

    XMLDocument doc;
    doc.LoadFile(filename_.c_str());

    auto element = doc.FirstChildElement("document");
    if(!element) {
        std::ostringstream oss;
        oss << "XML : file as no 'document' node."
            << "\n    file : " << filename_;
        throw std::runtime_error(oss.str());
    }

    element = element->FirstChildElement("chunk");
    if(!element) {
        std::ostringstream oss;
        oss << "XML : no chunk element in document element"
            << "\n    file : " << filename_;
        throw std::runtime_error(oss.str());
    }

    auto components = parse_components(element);
    for(const auto& cmp : components) {
        components_[cmp.id] = cmp;
    }

    element = element->FirstChildElement("cameras");
    if(!element) {
        std::ostringstream oss;
        oss << "XML : no cameras element in chunk element"
            << "\n    file : " << filename_;
        throw std::runtime_error(oss.str());
    }
    
    auto group = element->FirstChildElement("group");
    if(!group) {
        std::ostringstream oss;
        oss << "XML : no group element in cameras element"
            << "\n    file : " << filename_;
        throw std::runtime_error(oss.str());
    }

    unsigned int parsed = 0;
    while(group) {
        auto cam = group->FirstChildElement("camera");
        if(!cam) {
            std::ostringstream oss;
            oss << "XML : no camera element in group element"
                << "\n    file : " << filename_;
            throw std::runtime_error(oss.str());
        }

        while(cam) {
            auto info = parse_pose(cam);
            poseInfo_[info.sensorId].push_back(info);
            cam = cam->NextSiblingElement("camera");
            parsed++;
        }
        group = group->NextSiblingElement("group");
    }
    
    // Generating poses_ (compensate for metashape component transformation)
    for(const auto& info : poseInfo_) {
        poses_[info.first] = std::vector<Pose<float>>();
        auto it = poses_.find(info.first);
        for(const auto& pinfo : info.second) {
            auto p = pinfo.pose;
            const auto& cmp = components_.at(pinfo.componentId);
            p.translation() *= cmp.scale;
            it->second.push_back(cmp.transform*p);
        }
    }

    std::cout << "Parsed " << parsed << " camera elements." << std::endl;
}

} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::external::MetashapeOutput::Component& cmp)
{
    os << "Metashape component (id : " << cmp.id << ", label : " << cmp.label << ") :\n";
    os << "- transform : (scaling = " << cmp.scale << ")\n";
    {
        std::ostringstream oss;
        oss << cmp.transform.homogeneous_matrix();
        std::istringstream iss(oss.str());
        std::string token;
        while(std::getline(iss, token)) {
            os << "    " << token << "\n";
        }
    }
    os << "- region :\n"
       << "  - center   : " << cmp.regionCenter.transpose() << "\n"
       << "  - size     : " << cmp.regionSize.transpose()   << "\n"
       << "  - rotation :";
    {
        std::ostringstream oss;
        oss << cmp.regionRot;
        std::istringstream iss(oss.str());
        std::string token;
        std::getline(iss, token);
        os << token << "\n";
        while(std::getline(iss, token)) {
            os << "              " << token << "\n";
        }
    }
    return os;
}
