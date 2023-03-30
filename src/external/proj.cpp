#include <rtac_base/external/proj.h>

namespace rtac { namespace external {

ProjTransform::ProjTransform(const std::string& from, const std::string& to) :
    fromStr_(from), toStr_(to),
    context_(nullptr), transformation_(nullptr)
{
    context_ = proj_context_create();
    PJ* t = proj_create_crs_to_crs(context_, fromStr_.c_str(), toStr_.c_str(), nullptr);
    if(!t) {
        throw Exception("PROJ_ERROR") << " : could not create transformation "
                                      << fromStr_ << " -> " << toStr_;
    }
    transformation_ = proj_normalize_for_visualization(context_, t);
    if(!transformation_) {
        throw Exception("PROJ_ERROR") << " : could not normalize transformation "
                                      << fromStr_ << " -> " << toStr_;
    }
    proj_destroy(t);
}

ProjTransform::~ProjTransform()
{
    proj_destroy(transformation_);
    proj_context_destroy(context_);
}

PJ_COORD ProjTransform::forward(const PJ_COORD& from) const
{
    return proj_trans(transformation_, PJ_FWD, from);
}

PJ_COORD ProjTransform::backward(const PJ_COORD& from) const
{
    return proj_trans(transformation_, PJ_INV, from);
}

} //namespace external
} //namespace rtac

