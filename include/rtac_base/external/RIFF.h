#ifndef _DEF_RTAC_BASE_EXTERNAL_RIFF_H_
#define _DEF_RTAC_BASE_EXTERNAL_RIFF_H_

#include <vector>
#include <deque>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstring>
#include <fstream>

#include <rtac_base/Exception.h>

namespace rtac { namespace external { namespace riff {

struct RIFFError : public Exception
{
    RIFFError() : Exception("RTAC_RIFF_ERROR") {}
};

struct FourCC
{
    char data[4];

    FourCC()                         = default;
    FourCC(const FourCC&)            = default;
    FourCC& operator=(const FourCC&) = default;

    FourCC(const std::string& id) 
    {
        if(id.size() < 4) throw RIFFError() << " : Invalid FourCC " << id;
        std::memcpy(data, id.c_str(), 4);
    }

    std::string string()   const { return std::string(data, 4); }
    operator std::string() const { return this->string();       }

    bool operator==(const std::string& id) const {
        if(id.size() < 4) throw RIFFError() << " : Invalid FourCC " << id;
        return *this == id.c_str();
    }
    bool operator!=(const std::string& id) const { return !(*this == id);         }
    bool operator==(const FourCC& other)   const { return *this == other.data;    }
    bool operator!=(const FourCC& other)   const { return !(*this == other.data); }

    //protected: // dangerous

    bool operator==(const char* id) const {
        return data[0] == id[0]
            && data[1] == id[1]
            && data[2] == id[2]
            && data[3] == id[3];
    }
    bool operator!=(const char* id) const { return !(*this == id); }
};

struct ChunkHeader
{
    FourCC   id_;
    uint32_t size_;

    static ChunkHeader Unknown() { return ChunkHeader("UNKN",0); }
    static ChunkHeader End()     { return ChunkHeader("END ",0); }

    ChunkHeader()                              = default;
    ChunkHeader(const ChunkHeader&)            = default;
    ChunkHeader& operator=(const ChunkHeader&) = default;

    ChunkHeader(const std::string& id, uint32_t size) :
        id_(id), size_(size)
    {}

    FourCC   id()   const { return id_;   }
    uint32_t size() const { return size_; }

    bool is_end() const {
        return id_ == "END " && size_ == 0;
    }
};

struct FileHeader
{
    ChunkHeader header_;
    FourCC      formType_;

    FileHeader()                             = default;
    FileHeader(const FileHeader&)            = default;
    FileHeader& operator=(const FileHeader&) = default;

    FileHeader(const std::string& formType) :
        header_("RIFF", 0), formType_(formType)
    {}
    FileHeader(const FourCC& formType) :
        header_("RIFF", 0), formType_(formType)
    {}

    ChunkHeader header()    const { return header_;        }
    FourCC      id()        const { return header_.id();   }
    uint32_t    size()      const { return header_.size(); }
    FourCC      form_type() const { return formType_;      }
};

class Chunk
{
    protected:

    ChunkHeader header_;
    std::shared_ptr<uint8_t> data_; // data without header

    public:

    Chunk(const Chunk&)            = default;
    Chunk& operator=(const Chunk&) = default;

    Chunk(const ChunkHeader& header = ChunkHeader::Unknown()) :
        header_(header), data_(nullptr)
    {
        auto data = std::make_shared<std::vector<uint8_t>>(header_.size());
        data_ = std::shared_ptr<uint8_t>(data, data->data());
    }

    Chunk(const std::string& id, uint32_t size, const uint8_t* data) :
        Chunk(ChunkHeader(id, size))
    {
        std::memcpy(this->data(), data, size);
    }

    const ChunkHeader& header() const { return header_; }
    FourCC         id()   const { return header_.id();   }
    uint32_t       size() const { return header_.size(); }
    const uint8_t* data() const { return data_.get(); }
          uint8_t* data()       { return data_.get(); }
};

class RIFFReader
{
    protected:

    std::string   filename_;
    std::ifstream file_;

    FileHeader fileHeader_;
    ChunkHeader nextChunkHeader_;

    bool read_next_header();

    public:

    RIFFReader(const std::string& filename);

    FileHeader header()    const { return fileHeader_;             }
    uint32_t   size()      const { return fileHeader_.size();      }
    FourCC     form_type() const { return fileHeader_.form_type(); }
    
    ChunkHeader next_chunk_header() const { return nextChunkHeader_; }

    bool read_chunk(Chunk& chunk);
    ChunkHeader skip_chunk();
};

class RIFFWriter
{
    protected:

    std::string   filename_;
    std::ofstream file_;

    FileHeader fileHeader_;

    public:

    RIFFWriter(const std::string& filename, FourCC formType);
    ~RIFFWriter();

    FileHeader file_header() const { return fileHeader_;             }
    uint32_t   size()        const { return fileHeader_.size();      }
    FourCC     form_type()   const { return fileHeader_.form_type(); }
    
    void write_file_header();
    bool add_chunk(const Chunk& chunk, bool updateFileHeader = true);
};

} //namespace riff
} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::external::riff::FourCC& id);
std::ostream& operator<<(std::ostream& os, const rtac::external::riff::ChunkHeader& chunk);
std::ostream& operator<<(std::ostream& os, const rtac::external::riff::FileHeader& header);

#endif //_DEF_RTAC_BASE_EXTERNAL_RIFF_H_
