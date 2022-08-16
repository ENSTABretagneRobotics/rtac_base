#ifndef _DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_
#define _DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_

#include <iostream>
#include <vector>
#include <cstring>

#include <jpeglib.h>
#include <jerror.h>

#include <rtac_base/external/ImageCodecBase.h>

namespace rtac { namespace external {

class JPGCodec : public ImageCodecBase
{
    protected:

    FILE* file_;

    jpeg_error_mgr         err_;
    jpeg_decompress_struct info_;

    void clear();
    void reset();

    public:

    JPGCodec();
    ~JPGCodec();

    virtual void read_image(const std::string& path, bool invertRows = false);

};

}; //namespace external
}; //namespace rtac

#endif //_DEF_RTAC_BASE_EXTERNAL_JPG_CODEC_H_
