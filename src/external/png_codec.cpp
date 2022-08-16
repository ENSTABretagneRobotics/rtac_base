#include <rtac_base/external/png_codec.h>

#include <sstream>

namespace rtac { namespace external {

PNGCodec::PNGCodec() :
    ImageCodecBase(),
    handle_(nullptr),
    info_(nullptr),
    endInfo_(nullptr),
    file_(nullptr)
{}

PNGCodec::~PNGCodec()
{
    this->clear();
}

void PNGCodec::clear()
{
    if(handle_) {
        png_info** info(nullptr);
        png_info** endInfo(nullptr);
        if(info_) info = &info_;
        if(endInfo_) endInfo = &endInfo_;
        png_destroy_read_struct(&handle_, info, endInfo);
    }
    if(file_) {
        fclose(file_);
        file_ = nullptr;
    }

    width_    = 0;
    height_   = 0;
    step_     = 0;
    bitdepth_ = 0;
    channels_ = 0;
}

void PNGCodec::reset()
{
    this->clear();
    handle_ = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!handle_) {
        std::ostringstream oss;
        oss << "PNG error : could not allocate png_struct.";
        throw std::runtime_error(oss.str());
    }
    png_set_read_user_chunk_fn(handle_, this, &PNGCodec::read_chunk_callback_stub);
    png_set_error_fn(handle_, this,
                     &PNGCodec::png_error_callback,
                     &PNGCodec::png_warning_callback);

    info_ = png_create_info_struct(handle_);
    if(!info_) {
        std::ostringstream oss;
        oss << "PNG error : could not allocate png_info";
        throw std::runtime_error(oss.str());
    }
    endInfo_ = png_create_info_struct(handle_);
    if(!endInfo_) {
        std::ostringstream oss;
        oss << "PNG error : could not allocate png_info";
        throw std::runtime_error(oss.str());
    }
}

int PNGCodec::read_chunk_callback_stub(png_struct* handle, png_unknown_chunk* chunk)
{
    PNGCodec* codec = reinterpret_cast<PNGCodec*>(png_get_user_chunk_ptr(handle));
    if(!codec)
        return chunk->size;

    if(codec->handle() != handle) {
        std::ostringstream oss;
        oss << "Error on user chunk callback : png handles do not match.";
        throw std::runtime_error(oss.str());
    }

    return codec->read_chunk_callback(chunk);
}

void PNGCodec::png_error_callback(png_struct* handle, png_const_charp msg)
{
    auto codec = reinterpret_cast<PNGCodec*>(png_get_error_ptr(handle));
    std::ostringstream oss;
    oss << "PNG error : '" << msg << "'";
    throw std::runtime_error(oss.str());
}

void PNGCodec::png_warning_callback(png_struct* handle, png_const_charp msg)
{
    auto codec = reinterpret_cast<PNGCodec*>(png_get_error_ptr(handle));
    std::cerr << "PNG warning : " << msg << std::endl;
}

int PNGCodec::read_chunk_callback(const png_unknown_chunk* chunk)
{
    std::ostringstream oss;
    oss << chunk->name[0];
    oss << chunk->name[1];
    oss << chunk->name[2];
    oss << chunk->name[3];
    oss << chunk->name[4];

    std::cout << "Got unknown png chunk '" << oss.str()
              << "' (size " << chunk->size
              << " bytes)." << std::endl;

    //return -chunk->size; // chunk error
    //return           0;  // chunk not recognized
    return  chunk->size;   // ok.
}

void PNGCodec::read_image(const std::string& path, bool invertRows)
{
    // Have to reset entirely for reading multiple images because libpng not
    // clear on handle_ internal state.
    this->reset();

    file_ = fopen(path.c_str(), "rb");
    if(!file_) {
        std::ostringstream oss;
        oss << "Could not open .png file for reading : " << path;
        throw std::runtime_error(oss.str());
    }

    // ugly
    //if(setjmp(png_jmpbuf(handle_))) {
    //    std::ostringstream oss;
    //    oss << "PNG error occured.";
    //    throw std::runtime_error(oss.str());
    //}

    std::array<unsigned char, 8> header;
    if(fread(header.data(), 1, 8, file_) != 8) {
        std::ostringstream oss;
        oss << "Error reading png file signature : " << path;
        throw std::runtime_error(oss.str());
    }

    if(png_sig_cmp(header.data(), 0, 8)) {
        std::ostringstream oss;
        oss << "File does not seem to be a .png file : " << path;
        throw std::runtime_error(oss.str());
    }

    png_init_io(handle_, file_);
    png_set_sig_bytes(handle_, 8);

    png_read_info(handle_, info_);
    width_    = png_get_image_width (handle_, info_);
    height_   = png_get_image_height(handle_, info_);
    step_     = png_get_rowbytes    (handle_, info_);
    bitdepth_ = png_get_bit_depth   (handle_, info_);
    switch(png_get_color_type(handle_, info_)) {
        default:
            throw std::runtime_error("Unhandled .png color type");
            break;
        case PNG_COLOR_TYPE_GRAY:       channels_ = 1; break;
        case PNG_COLOR_TYPE_GRAY_ALPHA: channels_ = 2; break;
        //case PNG_COLOR_TYPE_PALETTE:    channels_ = 1; break;
        case PNG_COLOR_TYPE_RGB:        channels_ = 3; break;
        case PNG_COLOR_TYPE_RGB_ALPHA:  channels_ = 4; break;
    }
    
    data_.resize(height_*step_);
    std::vector<png_byte*> rows(height_);
    if(invertRows) {
        for(int h = 0; h < height_; h++)
            rows[h] = &data_[step_*(height_ - 1 - h)];
    }
    else {
        for(int h = 0; h < height_; h++)
            rows[h] = &data_[step_*h];
    }
    
    png_read_image(handle_, rows.data());
    png_read_end(handle_, endInfo_);

    fclose(file_);
    file_ = nullptr;
}



}; // namespace external
}; // namespace rtac

