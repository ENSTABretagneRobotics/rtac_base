#ifndef _DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_
#define _DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_

#include <iostream>
#include <vector>
#include <map>
#include <string>

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

    using PoseContainer = std::map<int, std::vector<PoseInfo>>;

    protected:

    std::string filename_;

    unsigned int poseCount_;
    PoseContainer poses_;

    public:

    MetashapeOutput() {}

    void load_file(const std::string& filename);
    void clear() { filename_ = ""; poses_.clear(); }

    const PoseContainer& poses()                  const { return poses_; }
    const std::vector<PoseInfo>& poses(int camId) const { return poses_.at(camId); }
};


} //namespace external
} //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_METASHAPE_H_
