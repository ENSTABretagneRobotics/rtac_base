#include <rtac_base/external/RIFF.h>
#include <fstream>

namespace rtac { namespace external { namespace riff {

RIFFReader::RIFFReader(const std::string& filename) :
    filename_(filename),
    file_(filename, std::ifstream::binary),
    fileHeader_("UNKN"),
    nextChunkHeader_(ChunkHeader::Unknown())
{
    if(!file_.is_open()) {
        throw RIFFError() << ": could not open file for reading '" << filename << "'";
    }
    if(!file_.read((char*)&fileHeader_, sizeof(fileHeader_))) {
        throw RIFFError() << " : could not read file header";
    }
    if(fileHeader_.id() != "RIFF") {
        std::cerr << "File does not have proper header (got '"
                  << fileHeader_.id() << "', expected 'RIFF')\n";
    }
    this->read_next_header();
}

bool RIFFReader::read_next_header()
{
    if(!file_.read((char*)&nextChunkHeader_, sizeof(nextChunkHeader_))) {
        nextChunkHeader_ = ChunkHeader("END ",0);
        return false;
    } 
    return true;
}

bool RIFFReader::read_chunk(Chunk& chunk)
{
    if(nextChunkHeader_.size() == 0) {
        return false;
    }

    chunk = Chunk(nextChunkHeader_);
    if(!file_.read((char*)chunk.data(), chunk.size())) {
        throw RIFFError() << " : could not read " << chunk.header()
                          << " in '" << filename_ << "'";
    }
    if(chunk.size() & 0x1) {
        file_.ignore(1);
    }
    this->read_next_header();

    return true;
}

ChunkHeader RIFFReader::skip_chunk()
{
    if(nextChunkHeader_.size() == 0) {
        return nextChunkHeader_;
    }
    uint32_t toSkip = nextChunkHeader_.size() + (nextChunkHeader_.size() & 0x1);
    if(!file_.seekg(toSkip, std::ifstream::cur)) {
        throw RIFFError() << " : error while skiping chunk in '" << filename_ << "'";
    }
    this->read_next_header();
    return nextChunkHeader_;
}

RIFFWriter::RIFFWriter(const std::string& filename, FourCC formType) :
    filename_(filename),
    file_(filename, std::ofstream::binary),
    fileHeader_(formType)
{
    if(!file_.is_open()) {
        throw RIFFError() << ": could not open file for writing '" << filename << "'";
    }
    this->write_file_header();
    file_.seekp(sizeof(FileHeader));
}

RIFFWriter::~RIFFWriter()
{
    this->write_file_header();
}

void RIFFWriter::write_file_header()
{
    auto lastPos = file_.tellp();
    file_.seekp(0);
    if(!file_.write((const char*)&fileHeader_, sizeof(FileHeader))) {
        throw RIFFError() << " : could not write header for file '" << filename_ << "'";
    }
    file_.seekp(lastPos);
}

bool RIFFWriter::add_chunk(const Chunk& chunk, bool updateFileHeader)
{
    uint64_t sizeToWrite = chunk.size() + sizeof(ChunkHeader)
                         +  (chunk.size() & 0x1); // must be 16bit aligned.

    if(sizeToWrite + fileHeader_.size() > 0xffffffff) {
        std::cerr << "Maximum size reached for RIFF file."
                  << " Cannot add chunk";
        return false;
    }

    if(!file_.write((const char*)&chunk.header(), sizeof(ChunkHeader))) {
        throw RIFFError() << " : could not write '" << chunk.id()
                          << "' chunk header in " << filename_;
    }
    if(!file_.write((const char*)chunk.data(), chunk.size())) {
        throw RIFFError() << " : could not write '" << chunk.id()
                          << "' chunk data in " << filename_;
    }
    if(chunk.size() & 0x1) {
        if(!file_.put(0)) {
            throw RIFFError() << " : could not pad '" << chunk.id()
                              << "' chunk data in " << filename_;
        }
    }

    fileHeader_.header_.size_ += sizeToWrite;
    if(updateFileHeader) {
        this->write_file_header();
    }

    return true;
}

} //namespace riff
} //namespace external
} //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::external::riff::FourCC& id)
{
    os.write(id.data, 4);
    return os;
}

std::ostream& operator<<(std::ostream& os, const rtac::external::riff::ChunkHeader& chunk)
{
    os << "Chunk '" << chunk.id() << "' (size : " << chunk.size() << ')';
    return os;
}

std::ostream& operator<<(std::ostream& os, const rtac::external::riff::FileHeader& header)
{
    os << header.header().id()
       << " file (size : " << header.size()
       << ", form type : " << header.form_type() << ")";
    return os;
}
