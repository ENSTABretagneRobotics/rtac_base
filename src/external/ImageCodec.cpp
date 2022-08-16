#include <rtac_base/external/ImageCodec.h>

#include <algorithm>

#ifdef RTAC_PNG
#include <rtac_base/external/png_codec.h>
#endif

#ifdef RTAC_JPEG
#include <rtac_base/external/jpg_codec.h>
#endif

namespace rtac { namespace external {

ImageCodecBase::ImageCodecBase() :
    width_(0),
    height_(0),
    step_(0),
    bitdepth_(0),
    channels_(0),
    data_(0)
{}


ImageCodec::ImageEncoding ImageCodec::encoding_from_extension(const std::string& path)
{
    // getting lowercase file extension
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if(ext == "png")       return PNG;
    else if(ext == "jpg")  return JPEG;
    else if(ext == "jpeg") return JPEG;
    else if(ext == "je")   return JPEG;
    else return UNKNOWN_ENCODING;
}

ImageCodec::ImageEncoding ImageCodec::find_encoding(const std::string& path)
{
    // Only encoding from extension for now.
    return encoding_from_extension(path);
}

ImageCodecBase::Ptr ImageCodec::create_codec(ImageEncoding encoding)
{
    if(encoding == UNKNOWN_ENCODING) {
        std::cerr << "Unknown image encoding. Cannot create codec." << std::endl;
        return nullptr;
    }

    if(encoding == PNG) {
        #ifdef RTAC_PNG
            return PNGCodec::Create();
        #else
            std::ostringstream oss;
            oss << "PNG file format is not supported. "
                   "Did you install libpng-dev before compiling rtac_base ?";
            throw std::runtime_error(oss.str);
        #endif
    }

    if(encoding == JPEG) {
        #ifdef RTAC_PNG
            return JPGCodec::Create();
        #else
            std::ostringstream oss;
            oss << "JPEG file format is not supported. "
                   "Did you install libijpeg8-dev before compiling rtac_base ?";
            throw std::runtime_error(oss.str);
        #endif
    }

    return nullptr;
}

ImageCodecBase::ConstPtr ImageCodec::read_image(const std::string& path, bool invertRows) const
{
    auto encoding = ImageCodec::find_encoding(path);
    if(encoding == UNKNOWN_ENCODING) {
        std::cerr << "Could not find encoding for file : " << path << std::endl;
        return nullptr;
    }

    if(codecs_.find(encoding) == codecs_.end()) {
        auto codec = ImageCodec::create_codec(encoding);
        if(!codec)
            return nullptr;
        codecs_[encoding] = codec;
    }

    codecs_[encoding]->read_image(path, invertRows);
    return codecs_[encoding];
}

}; //namespace external
}; //namespace rtac
