#include <rtac_base/external/metashape.h>

#include <sstream>

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
            poses_[info.sensorId].push_back(info);
            cam = cam->NextSiblingElement("camera");
            parsed++;
        }
        group = group->NextSiblingElement("group");
    }

    std::cout << "Parsed " << parsed << " camera elements." << std::endl;
}

} //namespace external
} //namespace rtac

