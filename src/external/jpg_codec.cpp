#include <rtac_base/external/jpg_codec.h>

#include <sstream>

namespace rtac { namespace external {

JPGCodec::JPGCodec() :
    ImageCodecBase(),
    file_(nullptr)
{
    bitdepth_ = BITS_IN_JSAMPLE;
}

JPGCodec::~JPGCodec()
{
    this->clear();
}

void JPGCodec::clear()
{
    //jpeg_destroy_decompress(&info_);
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

void JPGCodec::reset()
{
    this->clear();
}

void JPGCodec::read_image(const std::string& path, bool invertRows)
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

