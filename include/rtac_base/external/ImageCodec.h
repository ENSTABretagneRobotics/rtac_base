#ifndef _DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_
#define _DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_

#include <iostream>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_map>

namespace rtac { namespace external {

class ImageCodecBase;

/**
 * This class handles high level image codec manipulations.
 *
 * Features include :
 * - automatic codec selection from file path.
 */
class ImageCodec
{
    public:

    using CodecPtr      = std::shared_ptr<ImageCodecBase>;
    using CodecConstPtr = std::shared_ptr<const ImageCodecBase>;

    enum ImageEncoding {
        UNKNOWN_ENCODING,
        PNG,
        JPEG,
    };

    struct ImageInfo {
        std::size_t  width;
        std::size_t  height;
        std::size_t  step;
        unsigned int bitdepth;
        unsigned int channels;
    };

    static ImageEncoding encoding_from_extension(const std::string& path);
    static ImageEncoding find_encoding(const std::string& path);

    protected:

    mutable std::unordered_map<ImageEncoding, CodecPtr> codecs_;

    public:

    ImageCodec() {}

    static CodecPtr create_codec(ImageEncoding encoding);
    CodecPtr read_image(const std::string& path, bool invertRows = false) const;
};

/**
 * Base class for image decoder. Main purpose is to abstract the implementation
 * details for image codecs, and auto codec selection.
 */
class ImageCodecBase
{
    public:

    using Ptr      = std::shared_ptr<ImageCodecBase>;
    using ConstPtr = std::shared_ptr<const ImageCodecBase>;

    protected:

    std::size_t  width_;
    std::size_t  height_;
    std::size_t  step_;
    unsigned int bitdepth_;
    unsigned int channels_;
    std::vector<unsigned char> data_;

    ImageCodecBase();

    public:

    std::size_t  width()    const { return width_;  }
    std::size_t  height()   const { return height_; }
    std::size_t  step()     const { return step_;   }
    unsigned int bitdepth() const { return bitdepth_; }
    unsigned int channels() const { return channels_; }
    const std::vector<unsigned char>& data() const { return data_; }

    // Base to be implemented in a child class
    virtual void read_image(const std::string& path, bool invertRows) = 0;
    virtual void write_image(const std::string& path,
                             const ImageCodec::ImageInfo& info,
                             const unsigned char* data,
                             bool invertRows)
    {
        throw std::runtime_error("Not implemented");
    }
};

}; //namespace external
}; //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_
