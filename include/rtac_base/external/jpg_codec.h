#ifndef _DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_
#define _DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_

#include <iostream>
#include <vector>
#include <cstring>

#include <jpeglib.h>
#include <jerror.h>

namespace rtac { namespace external {

class JPGCodec
{
    protected:

    FILE* file_;

    std::size_t  width_;
    std::size_t  height_;
    std::size_t  step_;
    unsigned int channels_;
    std::vector<unsigned char> data_;

    jpeg_error_mgr         err_;
    jpeg_decompress_struct info_;

    void clear();
    void reset();

    public:

    JPGCodec();
    ~JPGCodec();

    void read_jpg(const std::string& path, bool invertRows = false);

    std::size_t width()  const { return width_;  }
    std::size_t height() const { return height_; }
    std::size_t step()   const { return step_;   }
    unsigned int channels() const { return channels_; }
    const std::vector<unsigned char>& data() const { return data_; }
};

inline JPGCodec::JPGCodec() :
    file_(nullptr),
    width_(0),
    height_(0),
    step_(0),
    channels_(0),
    data_(0)
{}

inline JPGCodec::~JPGCodec()
{
    this->clear();
}

inline void JPGCodec::clear()
{
    jpeg_destroy_decompress(&info_);
    std::memset(&info_, 0, sizeof(info_));
    std::memset(&err_,  0, sizeof(err_));

    if(file_) {
        fclose(file_);
        file_ = nullptr;
    }
    width_    = 0;
    height_   = 0;
    step_     = 0;
    channels_ = 0;
}

inline void JPGCodec::reset()
{
    this->clear();
}

inline void JPGCodec::read_jpg(const std::string& path, bool invertRows)
{
    this->reset();

    file_ = fopen(path.c_str(), "rb");
    if(!file_) {
        std::ostringstream oss;
        oss << "Could not open .jpg file for reading : " << path;
        throw std::runtime_error(oss.str());
    }

    info_.err = jpeg_std_error(&err_);
    jpeg_create_decompress(&info_);
    
    jpeg_stdio_src(&info_,  file_);
    jpeg_read_header(&info_, TRUE);
    jpeg_start_decompress(&info_);

    width_    = info_.output_width;
    height_   = info_.output_height;
    channels_ = info_.num_components;
    step_     = width_*channels_; // assuming packed pixels

    data_.resize(height_*step_);
    std::vector<unsigned char*> rows(height_);
    if(invertRows) {
        for(int h = 0; h < height_; h++)
            rows[h] = &data_[step_*(height_ - 1 - h)];
    }
    else {
        for(int h = 0; h < height_; h++)
            rows[h] = &data_[step_*h];
    }
    
    // Have to read line by line because libjpeg internal memory may not handle
    // a full image.
    for(int i = 0; info_.output_scanline < info_.output_height; i++) {
        jpeg_read_scanlines(&info_, &rows[i], 1);
    }
    
    jpeg_finish_decompress(&info_);
    fclose(file_);
    file_ = nullptr;
}

}; //namespace external
}; //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_
