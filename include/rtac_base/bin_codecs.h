#ifndef _DEF_RTAC_BASE_BIN_CODECS_H_
#define _DEF_RTAC_BASE_BIN_CODECS_H_

#include <stdexcept>

namespace rtac {

void hex_encode(unsigned char* dst, const unsigned char* src, std::size_t inputSize);
void hex_decode(unsigned char* dst, const unsigned char* src, std::size_t inputSize);

template <typename T> inline
void hex_encode(unsigned char* dst, const T* src, std::size_t inputSize)
{
    hex_encode(dst, (const unsigned char*)src, sizeof(T)*inputSize);
}

template <typename T> inline
void hex_decode(T* dst, const unsigned char* src, std::size_t inputSize)
{
    if(inputSize % (2*sizeof(T)) != 0) {
        throw std::runtime_error(
            "rtac::hex_decode : inputSize not a multiple of target decoding type.");
    }
    hex_decode((unsigned char*)dst, src, inputSize);
}

template <typename T,     template<typename> class VectorT,
          typename CharT, template<typename> class VectorCharT>
inline void hex_encode(VectorCharT<CharT>& output, const VectorT<T>& input)
{
    static_assert(sizeof(CharT) == 1);
    output.resize(2*sizeof(T)*input.size());
    hex_encode((unsigned char*)output.data(), input.data(), input.size());
}

template <typename T,     template<typename> class VectorT,
          typename CharT, template<typename> class VectorCharT>
inline void hex_decode(VectorT<T>& output, const VectorCharT<CharT>& input)
{
    static_assert(sizeof(CharT) == 1);
    output.resize(input.size() / (2 * sizeof(T)));
    hex_decode(output.data(), (const unsigned char*)input.data(), input.size());
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_BIN_CODECS_H_
