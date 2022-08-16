#ifndef _DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_
#define _DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_

#include <iostream>
#include <cstring>
#include <vector>
#include <memory>
#include <unordered_map>

namespace rtac { namespace external {

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

    std::size_t  width()  const { return width_;  }
    std::size_t  height() const { return height_; }
    std::size_t  step()   const { return step_;   }
    unsigned int bitdepth() const { return bitdepth_; }
    unsigned int channels() const { return channels_; }
    const std::vector<unsigned char>& data() const { return data_; }

    // Base to be implemented in a child class
    virtual void read_image(const std::string& path, bool invertRows) = 0;
};

/**
 * This class handles high level image codec manipulations.
 *
 * Features include :
 * - automatic codec selection from file path.
 */
class ImageCodec
{
    public:

    using CodecPtr = ImageCodecBase::Ptr;

    enum ImageEncoding {
        UNKNOWN_ENCODING,
        PNG,
        JPEG,
    };

    static ImageEncoding encoding_from_extension(const std::string& path);
    static ImageEncoding find_encoding(const std::string& path);

    protected:

    mutable std::unordered_map<ImageEncoding, CodecPtr> codecs_;

    public:

    ImageCodec() {}

    static ImageCodecBase::Ptr create_codec(ImageEncoding encoding);
    ImageCodecBase::ConstPtr  read_image(const std::string& path,
                                         bool invertRows = false) const;
};

}; //namespace external
}; //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_IMAGE_CODEC_BASE_H_
