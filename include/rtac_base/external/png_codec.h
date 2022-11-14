#ifndef _DEF_RTAC_BASE_EXTERNAL_PNG_CODEC_H_
#define _DEF_RTAC_BASE_EXTERNAL_PNG_CODEC_H_

#include <iostream>
#include <array>
#include <vector>
#include <type_traits>

#include <png.h>

#include <rtac_base/types/Point.h>

#include <rtac_base/external/ImageCodec.h>

namespace rtac { namespace external {

template <unsigned int D>
struct PNGPixelError : std::false_type {};

template <unsigned int Depth>
struct PNGScalar {
    static_assert(PNGPixelError<Depth>::value, "Unhandled bit depth (must be 8 or 16)");
};
template<> struct PNGScalar<8>  { using type = unsigned char; };
template<> struct PNGScalar<16> { using type = uint16_t;      };

template <unsigned int Channels, typename T>
struct PNGColorType {
    static_assert(PNGPixelError<Channels>::value, "Unhandled bit depth (must be 1,2,3,4)");
};
template<typename T> struct PNGColorType<2,T> { using type = types::Point2<T>; };
template<typename T> struct PNGColorType<3,T> { using type = types::Point3<T>; };
template<typename T> struct PNGColorType<4,T> { using type = types::Point4<T>; };

template <unsigned int ChannelCount, unsigned int ChannelDepth>
struct PNGPixelType {
    using Scalar = typename PNGScalar<ChannelDepth>::type;
    using type = std::conditional<ChannelCount == 1, Scalar,
        PNGColorType<ChannelCount, Scalar>>;
};

class PNGCodec : public ImageCodecBase
{
    public:

    using Ptr      = std::shared_ptr<PNGCodec>;
    using ConstPtr = std::shared_ptr<const PNGCodec>;
    
    protected:
    
    png_struct* handle_;
    png_info*   info_;
    png_info*   endInfo_;
    FILE* file_;
    
    static  int read_chunk_callback_stub(png_struct* handle, png_unknown_chunk* chunk);
    virtual int read_chunk_callback(const png_unknown_chunk* chunk);

    static void png_error_callback(png_struct* handle, png_const_charp msg);
    static void png_warning_callback(png_struct* handle, png_const_charp msg);

    void clear();
    void reset_read();
    void reset_write();

    public:

    PNGCodec();
    ~PNGCodec();

    static Ptr Create() { return Ptr(new PNGCodec()); }

    virtual void read_image(const std::string& path, bool invertRows = false);
    virtual void write_image(const std::string& path,
                             const ImageCodec::ImageInfo& info,
                             const unsigned char* data,
                             bool invertRows);

    const png_struct* handle()   const { return handle_;  }
    const png_info*   info()     const { return info_;    }
    const png_info*   end_info() const { return endInfo_; }
};

}; //namespace external
}; //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_PNG_CODEC_H_
