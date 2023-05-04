#ifndef _DEF_RTAC_BASE_SERIALIZATION_SONAR_PING_H_
#define _DEF_RTAC_BASE_SERIALIZATION_SONAR_PING_H_

#include <iosfwd>

#include <rtac_base/common.h>
#include <rtac_base/types/TypeInfo.h>
#include <rtac_base/types/SonarPing.h>
#include <rtac_base/containers/HostVector.h>
#include <rtac_base/serialization/serialization.h>

namespace rtac {

#pragma pack(push,1)
struct PingInfo
{
    StringId<4> type; // "P1D_" or "P2D_"
    uint32_t    rangeCount;
    float       rangeMin;
    float       rangeMax;
    uint32_t    bearingCount;
    float       bearingMin;
    float       bearingMax;
    ScalarId    scalarType;

    Linspace<float> ranges() const {
        return Linspace<float>(rangeMin, rangeMax, rangeCount);
    }
    uint64_t size() const { return rangeCount * bearingCount; }
};
#pragma pack(pop)

/**
 * This serialization scheme is following the RIFF specification.
 *
 * (It does not generates the file header)
 */
template <typename T> inline
std::ostream& serialize(std::ostream& os, const Ping2D<T, HostVector>& ping)
{
    PingInfo info;
    info.type          = StringId<4>("P2D_");
    info.rangeCount    = ping.range_count();
    info.rangeMin      = ping.range_min();
    info.rangeMax      = ping.range_max();
    info.bearingCount  = ping.bearing_count();
    info.bearingMin    = ping.bearing_min();
    info.bearingMax    = ping.bearing_max();
    info.scalarType    = GetScalarId<T>::value;

    os << ChunkHeader32("PINF", sizeof(PingInfo)) << (const char*)&info; 
    os << ChunkHeader32("PDAT", sizeof(T)*ping.size())
       << (const char*)ping.ping_data();
    os << ChunkHeader32("BDAT", sizeof(float)*ping.bearings().size())
       << (const char*)ping.bearings().data();

    return os;
}

template <typename T, template<typename>class VectorT> inline
std::ostream& serialize(std::ostream& os, const Ping2D<T, VectorT>& ping)
{
    return serialize(os, Ping2D<T,HostVector>(ping));
}

/**
 * This deserialization scheme is following the RIFF specification, except that
 * chunks must always be in the proper order.
 *
 * It does not skip the file header.
 */
template <typename T> inline
std::istream& deserialize(std::istream& is, Ping2D<T, HostVector>& ping)
{
    auto header = ChunkHeader32::Empty();
    bool infoLoaded     = false;
    bool pingDataLoaded = false;
    bool bearingsLoaded = false;

    PingInfo          info;
    HostVector<T>     pingData;
    HostVector<float> bearingData;

    auto chunks_loaded = [&]() {
        return infoLoaded && pingDataLoaded && bearingsLoaded;
    };
    while(is && !chunks_loaded())
    {
        auto stub = is >> header;
        if(!is) break;

        if(header.id == "PINF") {
            stub >> info;
            infoLoaded = true;
            continue;
        }
        if(header.id == "PDAT") {
            stub >> pingData;
            pingDataLoaded = true;
            continue;
        }
        if(header.id == "BDAT") {
            stub >> bearingData;
            bearingsLoaded = true;
            continue;
        }
    }
    if(!chunks_loaded()) {
        throw SerializeError() << " : not all data where loaded for Ping2D serialization.";
    }

    if(pingData.size() != info.size() || bearingData.size() != info.bearingCount) {
        throw SerializeError() << " : inconsistent sizes for ping deserialization.";
    }
    if(info.scalarType != GetScalarId<T>::value) {
        throw SerializeError() << " : wrong type for decoding (user requested "
                               << to_string(GetScalarId<T>::value) << ", got "
                               << to_string(info.scalarType) << ')';
    }

    ping = std::move(Ping2D<T,HostVector>(info.ranges(),
                                          std::move(bearingData),
                                          std::move(pingData)));
    
    return is;
}

template <typename T, template<typename>class VectorT> inline
std::istream& deserialize(std::istream& is, Ping2D<T, VectorT>& ping)
{
    Ping2D<T,HostVector> tmp;
    deserialize(is, tmp);
    ping = tmp;
    return is;
}

}// namespace rtac

#endif //_DEF_RTAC_BASE_SERIALIZATION_SONAR_PING_H_
