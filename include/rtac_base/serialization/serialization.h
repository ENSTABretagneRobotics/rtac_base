#ifndef _DEF_RTAC_BASE_SERIALIZATION_SERIALIZATION_H_
#define _DEF_RTAC_BASE_SERIALIZATION_SERIALIZATION_H_

#include <iosfwd>
#include <cstdint>
#include <string>

#include <rtac_base/common.h>
#include <rtac_base/Exception.h>
#include <rtac_base/types/StringId.h>
#include <rtac_base/containers/HostVector.h>


namespace rtac {
struct ChunkHeader32;
struct ChunkHeader64;
struct SerializeStub;
struct DeserializeStub32;
struct DeserializeStub64;
} 
inline rtac::SerializeStub     operator<<(std::ostream&, const rtac::ChunkHeader32&);
inline rtac::SerializeStub     operator<<(std::ostream&, const rtac::ChunkHeader64&);
inline rtac::DeserializeStub32 operator>>(std::istream&, rtac::ChunkHeader32&);
inline rtac::DeserializeStub64 operator>>(std::istream&, rtac::ChunkHeader64&);

namespace rtac {

struct SerializeError : public Exception
{
    SerializeError() : Exception("RTAC_SERIALIZE_ERROR") {}
};

#pragma pack(push,1)
/**
 * This is a header for binary representation of serialized objects. It
 * contains a 4 bytes identifier followed by a 4 bytes size of the following
 * data.
 */
struct ChunkHeader32
{
    StringId<4> id;
    uint32_t    size;

    ChunkHeader32()                                = default;
    ChunkHeader32(const ChunkHeader32&)            = default;
    ChunkHeader32& operator=(const ChunkHeader32&) = default;

    ChunkHeader32(const StringId<4>& i, uint32_t s) : id(i), size(s) {}
    ChunkHeader32(const std::string& i, uint32_t s) : id(i), size(s) {}
    static ChunkHeader32 Empty() { return ChunkHeader32("UNKN", 0); }
    
    // saved size is aligned on 32bits
    uint32_t written_size() const {
        if(size == 0) {
            return 0;
        }
        return 4*(((size - 1) / 4) + 1);
    }
    uint64_t padding() const { return written_size() - size; }
};

/**
 * This is a header for binary representation of serialized objects. It
 * contains a 8 bytes identifier followed by a 8 bytes size of the following
 * data.
 */
struct ChunkHeader64
{
    StringId<8> id;
    uint64_t    size;

    ChunkHeader64()                                = default;
    ChunkHeader64(const ChunkHeader64&)            = default;
    ChunkHeader64& operator=(const ChunkHeader64&) = default;
    static ChunkHeader64 Empty() { return ChunkHeader64("UNKN", 0); }

    ChunkHeader64(const StringId<8>& i, uint64_t s) : id(i), size(s) {}
    ChunkHeader64(const std::string& i, uint64_t s) : id(i), size(s) {}
    
    // saved size is aligned on 64bits
    uint64_t written_size() const {
        if(size == 0) {
            return 0;
        }
        return 8*(((size - 1) / 8) + 1);
    }
    uint64_t padding() const { return written_size() - size; }
};
#pragma pack(pop)

/**
 * This is a temporary type automatically instanciated when using operator<< on
 * a std::ostream and a ChunkHeader.
 *
 * This type is mainly about syntactic sugar.
 */
class SerializeStub
{
    public:
    
    //restricting this object instanciation to this function
    friend SerializeStub (::operator<<)(std::ostream&, const rtac::ChunkHeader32&);
    friend SerializeStub (::operator<<)(std::ostream&, const rtac::ChunkHeader64&);

    protected:

    std::ostream& os_;
    uint64_t      dataSize_;
    uint64_t      padding_;

    SerializeStub(std::ostream& os, const ChunkHeader32& header) :
        os_(os), 
        dataSize_(header.size),
        padding_(header.padding())
    {
        os_.write((const char*)&header, sizeof(ChunkHeader32));
    }
    SerializeStub(std::ostream& os, const ChunkHeader64& header) :
        os_(os), 
        dataSize_(header.size),
        padding_(header.padding())
    {
        os_.write((const char*)&header, sizeof(ChunkHeader64));
    }

    public:

    std::ostream& operator<<(const char* data) {
        os_.write(data, dataSize_);
        for(unsigned int i = 0; i < padding_; i++) os_.put(0);
        return os_;
    }
};

/**
 * This is a temporary type automatically instanciated when using operator>> on
 * a std::ostream and an empty ChunkHeader.
 */
class DeserializeStub32
{
    public:
    
    //restricting this object instanciation to this function
    friend DeserializeStub32 (::operator>>)(std::istream&, rtac::ChunkHeader32&);

    protected:

    std::istream&        is_;
    rtac::ChunkHeader32& header_;

    DeserializeStub32(std::istream& is, ChunkHeader32& header) :
        is_(is), 
        header_(header)
    {
        is_.read((char*)&header_, sizeof(ChunkHeader32));
    }

    public:

    template <typename T>
    std::istream& operator>>(T& item) {
        if(header_.size != sizeof(item)) {
            throw SerializeError() << " : wrong size for serialized data";
        }
        is_.read((char*)&item, sizeof(item));
        is_.ignore(header_.padding());

        return is_;
    }

    template <typename T>
    std::istream& operator>>(HostVector<T>& out) {
        out.resize(header_.size / sizeof(T));
        unsigned int remainer = header_.size - out.size()*sizeof(T);
        if(remainer != 0) {
            std::cerr << "Warning : serialized data size is not a multiple of "
                      << "requested type size.\n";
        }
        is_.read((char*)out.data(), sizeof(T)*out.size());
        is_.ignore(remainer + header_.padding());

        return is_;
    }
};
/**
 * This is a temporary type automatically instanciated when using operator>> on
 * a std::ostream and an empty ChunkHeader.
 */
class DeserializeStub64
{
    public:
    
    //restricting this object instanciation to this function
    friend DeserializeStub64 (::operator>>)(std::istream&, rtac::ChunkHeader64&);

    protected:

    std::istream&        is_;
    rtac::ChunkHeader64& header_;

    DeserializeStub64(std::istream& is, ChunkHeader64& header) :
        is_(is), 
        header_(header)
    {
        is_.read((char*)&header_, sizeof(ChunkHeader64));
    }

    public:

    template <typename T>
    std::istream& operator>>(HostVector<T>& out) {
        out.resize(header_.size / sizeof(T));
        unsigned int remainer = header_.size - out.size()*sizeof(T);
        if(remainer != 0) {
            std::cerr << "Warning : serialized data size is not a multiple of "
                      << "requested type size.\n";
        }
        is_.read((char*)out.data(), sizeof(T)*out.size());
        is_.ignore(remainer + header_.padding());

        return is_;
    }
};

} //namespace rtac


inline rtac::SerializeStub operator<<(std::ostream& os, const rtac::ChunkHeader32& chunkHeader)
{
    return rtac::SerializeStub(os, chunkHeader);
}

inline rtac::SerializeStub operator<<(std::ostream& os, const rtac::ChunkHeader64& chunkHeader)
{
    return rtac::SerializeStub(os, chunkHeader);
}

inline rtac::DeserializeStub32 operator>>(std::istream& is, rtac::ChunkHeader32& chunkHeader)
{
    return rtac::DeserializeStub32(is, chunkHeader);
}

inline rtac::DeserializeStub64 operator>>(std::istream& is, rtac::ChunkHeader64& chunkHeader)
{
    return rtac::DeserializeStub64(is, chunkHeader);
}

#endif //_DEF_RTAC_BASE_SERIALIZATION_SERIALIZATION_H_

