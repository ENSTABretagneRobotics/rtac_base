#ifndef _DEF_RTAC_BASE_SONAR_PING_H_
#define _DEF_RTAC_BASE_SONAR_PING_H_

#include <string>

#include <rtac_base/common.h>
#include <rtac_base/types/TypeInfo.h>
#include <rtac_base/types/SonarPing.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/external/RIFF.h>

namespace rtac {

RTAC_PACKED_STRUCT( RIFFPingInfo2D,
    uint32_t rangeCount;
    float    rangeMin;
    float    rangeMax;
    uint32_t bearingCount;
    float    bearingMin;
    float    bearingMax;
    ScalarId scalarType;

    Linspace<float> ranges() const {
        return Linspace<float>(rangeMin, rangeMax, rangeCount);
    }
);

template <typename T> inline
bool save_as_riff(const std::string& filename, const Ping2D<T, HostVector>& ping)
{
    external::riff::RIFFWriter file(filename, {"SON2"});
    return save_in(file, ping);
}

template <typename T> inline
bool save_in(external::riff::RIFFWriter& file, const Ping2D<T, HostVector>& ping)
{
    RIFFPingInfo2D info;
    info.rangeCount    = ping.range_count();
    info.rangeMin      = ping.range_min();
    info.rangeMax      = ping.range_max();
    info.bearingCount  = ping.bearing_count();
    info.bearingMin    = ping.bearing_min();
    info.bearingMax    = ping.bearing_max();
    info.scalarType    = GetScalarId<T>::value;

    file.add_chunk(external::riff::Chunk("INF2",
                                         sizeof(RIFFPingInfo2D),
                                         (const uint8_t*)&info));
    file.add_chunk(external::riff::Chunk("PDAT",
                                         sizeof(T)*ping.size(),
                                         (const uint8_t*)ping.ping_data()));
    file.add_chunk(external::riff::Chunk("BDAT",
                                         sizeof(float)*ping.bearing_count(),
                                         (const uint8_t*)ping.bearings().data()));
    return true;
}

class PingParser2D
{
    protected:

    Chunk infoChunk_;
    Chunk pingChunk_;
    Chunk bearingChunk_;

    RIFFPingInfo2D info_;

    public:

    PingParser2D(const Chunk& infoChunk,
                 const Chunk& pingChunk,
                 const Chunk& bearingChunk) :
        infoChunk_(infoChunk),
        pingChunk_(pingChunk),
        bearingChunk_(bearingChunk)
    {
        if(infoChunk_.size() < sizeof(info_) {
            throw external::RIFFError()
                << " : not enough data in 'INF2' chunk. Ping file may be corrupted";
        }
        std::memcpy(&info_, infoChunk_.data(), sizeof(info_));
    }

    template <typename T,
              template<typename>class VectorT>
    Ping2D<T,VectorT> build_ping()
    {
        if(GetScalarId<T>::value != info_.scalarType) {
            throw external::RIFFError()
                << " : wrong type for ping file parsing. (user asked "
                << to_string(GetScalarId<T>::value) << ", file is "
                << to_string(info_.scalarType) << ')';
        }
        unsigned int pingSize = info_.rangeCount * info_.bearingCount;
        if(pingChunk_.size() < sizeof(T)*pingSize) {
            throw external::RIFFError()
                << " : not enough data in 'PDAT' chunk. Ping file may be corrupted";
        }
        if(bearingChunk_.size() < sizeof(float)*info_.bearingCount) {
            throw external::RIFFError()
                << " : not enough data in 'BDAT' chunk. Ping file may be corrupted";
        }

        HostVector<T> pingData(info_.rangeCount * info_.bearingCount);
        std::memcpy(pingData.data(),
                    pingChunk_.data(),
                    sizeof(T)*pingData.size());

        HostVector<float> bearings(info_.bearingCount);
        std::memcpy(bearings.data(),
                    bearingChunk_.data(), 
                    sizeof(float)*bearings.size());
        return Ping2D<T,VectorT>(info_.ranges(), bearings, pingData);
    }
};

template <typename T> inline
PingParser2D load_ping2d_from_riff(const std::string& filename)
{
    external::riff::RIFFReader reader(filename);

    Chunk infoChunk;
    Chunk pingChunk;
    Chunk bearingChunk;

    while(!reader.next_chunk_header().is_end()) {
        if(reader.next_chunk_header().id() == "INF2") {
            reader.read_chunk(infoChunk);
        }
        else if(reader.next_chunk_header().id() == "PDAT") {
            reader.read_chunk(pingChunk);
        }
        else if(reader.next_chunk_header().id() == "BDAT") {
            reader.read_chunk(bearingChunk);
        }
    }

    return PingParser2D(infoChunk, pingChunk, bearingChunk);
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_SONAR_PING_H_
