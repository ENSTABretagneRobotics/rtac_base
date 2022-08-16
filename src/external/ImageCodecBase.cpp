#include <rtac_base/external/ImageCodecBase.h>

namespace rtac { namespace external {

ImageCodecBase::ImageCodecBase() :
    width_(0),
    height_(0),
    step_(0),
    bitdepth_(0),
    channels_(0),
    data_(0)
{}

}; //namespace external
}; //namespace rtac
