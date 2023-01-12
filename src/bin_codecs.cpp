#include <rtac_base/bin_codecs.h>

namespace rtac {

unsigned char to_hex(unsigned char value)
{
    value = value & 0x0f;
    return (value < 10) ? value + (unsigned char)'0' : value + (unsigned char)'A';
}

unsigned char from_hex(unsigned char value)
{
    return (value < 'A') ? value - (unsigned char)'0' : value - (unsigned char)'A';
}

void hex_encode(unsigned char* dst, const unsigned char* src, std::size_t inputSize)
{
    for(std::size_t i = 0; i < inputSize; i++) {
        dst[0] = to_hex(0x0f & (*src));
        dst[1] = to_hex((*src) >> 4);
        dst += 2;
        src += 1;
    }
}

void hex_decode(unsigned char* dst, const unsigned char* src, std::size_t inputSize)
{
    for(std::size_t i = 0; i < inputSize / 2; i++) {
        *dst = from_hex(src[0]) + (from_hex(src[1]) << 4);
        dst += 1;
        src += 2;
    }
}

} //namespace rtac
