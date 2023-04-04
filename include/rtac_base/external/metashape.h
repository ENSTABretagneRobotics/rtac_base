#ifndef _DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_
#define _DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <deque>

#include <rtac_base/types/Pose.h>

namespace rtac { namespace external {

class MetashapeOutput
{
    public:

    struct PoseInfo
    {
        Pose<float> pose;
        int id;
        int sensorId;
        int componentId;
        std::string label;
    };

    struct Component
    {
        using Vec3 = Pose<float>::Vec3;
        using Mat3 = Pose<float>::Mat3;

        int id;
        std::string label;

        Pose<float> transform;
        float       scale;
        
        Vec3 regionCenter;
        Vec3 regionSize;
        Mat3 regionRot;

        Component() :
            id(0), label("NoLabel"), scale(1.0f),
            regionCenter({0,0,0}),
            regionSize({1,1,1}),
            regionRot(Mat3::Identity())
        {}
    };

    using ComponentContainer = std::map<int, Component>;
    using PoseInfoContainer  = std::map<int, std::vector<PoseInfo>>;
    using PoseContainer      = std::map<int, std::vector<Pose<float>>>;

    protected:
    
    std::string filename_;

    ComponentContainer components_;

    unsigned int poseCount_;
    PoseInfoContainer poseInfo_;

    PoseContainer poses_;

    public:

    MetashapeOutput() {}

    void load_file(const std::string& filename);
    void clear() { filename_ = ""; poseInfo_.clear(); }

    const ComponentContainer& components() const { return components_; }
    const Component& component(int id) const { return components_.at(id); }

    const PoseInfoContainer&     pose_info()          const { return poseInfo_; }
    const std::vector<PoseInfo>& pose_info(int camId) const { return poseInfo_.at(camId); }

    const PoseContainer&            poses()          const { return poses_; }
    const std::vector<Pose<float>>& poses(int camId) const { return poses_.at(camId); }

};

} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::external::MetashapeOutput::Component& cmp);

#endif //_DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_
