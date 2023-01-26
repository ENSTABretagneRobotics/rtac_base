#ifndef _DEF_RTAC_BASE_KALIBR_H_
#define _DEF_RTAC_BASE_KALIBR_H_

#include <iostream>
#include <fstream>

#include <Eigen/Dense>

#include <yaml-cpp/yaml.h>

namespace rtac { namespace external {


class KalibrCalibration
{
    public:

    using Mat4 = Eigen::Matrix4d;
    struct CameraInfo
    {
        using Mat4 = Eigen::Matrix4d;

        std::string  name;
        unsigned int index;
        unsigned int width;
        unsigned int height;
    
        std::string cameraModel;
        std::vector<double> intrinsics;
    
        std::string distortionModel;
        std::vector<double> distortionCoeffs;

        std::vector<unsigned int> camOverlaps;

        // this transform is relative to the index - 1 camera.
        // Is Identity for the first one
        Mat4 transform;
    };

    protected:
    
    std::string filename_;
    std::string content_;

    std::vector<CameraInfo> cameras_;

    void parse();
    void add_camera(const std::string& name, const YAML::Node& node);

    public:

    KalibrCalibration(const std::string& calibrationFile);

    const std::vector<CameraInfo>& cameras() const { return cameras_; }
};

} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os,
                         const rtac::external::KalibrCalibration::CameraInfo& info);
std::ostream& operator<<(std::ostream& os, const rtac::external::KalibrCalibration& calib);

#endif //_DEF_RTAC_BASE_KALIBR_H_
