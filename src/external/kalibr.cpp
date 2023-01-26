#include <rtac_base/external/kalibr.h>

namespace rtac { namespace external {

KalibrCalibration::KalibrCalibration(const std::string& calibrationFile) :
    filename_(calibrationFile)
{
    std::ostringstream oss;
    std::ifstream f(filename_, std::ifstream::in);
    std::string line;
    while(std::getline(f, line)) {
        oss << line << std::endl;
    }
    content_ = oss.str();
    this->parse();
}

void KalibrCalibration::add_camera(const std::string& name, const YAML::Node& node)
{
    CameraInfo info;

    info.name   = name;
    info.index  = cameras_.size();
    info.width  = node["resolution"][0].as<unsigned int>();
    info.height = node["resolution"][1].as<unsigned int>();

    info.cameraModel = node["camera_model"].as<std::string>();
    info.intrinsics  = node["intrinsics"].as<std::vector<double>>();

    info.distortionModel  = node["distortion_model"].as<std::string>();
    info.distortionCoeffs = node["distortion_coeffs"].as<std::vector<double>>();

    info.camOverlaps = node["cam_overlaps"].as<std::vector<unsigned int>>();

    auto tnode = node["T_cn_cnm1"];
    if(!tnode.IsDefined()) {
        info.transform = Mat4::Identity();
    }
    else {
        unsigned int i = 0;
        for(auto v : tnode[0].as<std::vector<double>>()) {
            info.transform(0,i++) = v;
        }
        i = 0;
        for(auto v : tnode[1].as<std::vector<double>>()) {
            info.transform(1,i++) = v;
        }
        i = 0;
        for(auto v : tnode[2].as<std::vector<double>>()) {
            info.transform(2,i++) = v;
        }
        i = 0;
        for(auto v : tnode[3].as<std::vector<double>>()) {
            info.transform(3,i++) = v;
        }
    }

    cameras_.push_back(info);
}

void KalibrCalibration::parse()
{
    YAML::Node root = YAML::Load(content_);

    for(const auto& node : root) {
        this->add_camera(node.first.as<std::string>(), node.second);
    }
}

} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os, 
    const rtac::external::KalibrCalibration::CameraInfo& info)
{
    os << "CameraInfo : " << info.name
       << "\n- index             : " << info.index
       << "\n- resolution        : " << info.width << 'x' << info.height
       << "\n- camera model      : " << info.cameraModel
       << "\n- intrinsics        :"; for(auto v : info.intrinsics) os << ' ' << v;
    os << "\n- distortion model  : " << info.distortionModel
       << "\n- distortion coeffs :"; for(auto v : info.distortionCoeffs) os << ' ' << v;
    os << "\n- cam overlaps      :"; for(auto v : info.camOverlaps) os << ' ' << v;
    os << "\n- transform ";
    if(info.index > 0)  os << info.index-1 << "<-" << info.index << " :";
    if(info.index == 0) os << "0<-0 :";
    std::ostringstream oss;
    oss << info.transform;
    std::istringstream iss(oss.str());
    std::string line;
    while(std::getline(iss, line)) {
        os << "\n    " << line;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, const rtac::external::KalibrCalibration& calib)
{
    for(auto c : calib.cameras()) {
        os << c << std::endl;
    }
    return os;
}
