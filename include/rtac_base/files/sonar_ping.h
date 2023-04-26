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
    uint8_t  dataIsComplex;
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
    info.dataIsComplex = 0;

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

template <typename T> inline
bool save_in(external::riff::RIFFWriter& file, const Ping2D<Complex<T>, HostVector>& ping)
{
    RIFFPingInfo2D info;
    info.rangeCount    = ping.range_count();
    info.rangeMin      = ping.range_min();
    info.rangeMax      = ping.range_max();
    info.bearingCount  = ping.bearing_count();
    info.bearingMin    = ping.bearing_min();
    info.bearingMax    = ping.bearing_max();
    info.scalarType    = GetScalarId<T>::value;
    info.dataIsComplex = 1;

    file.add_chunk(external::riff::Chunk("INF2",
                                         sizeof(RIFFPingInfo2D),
                                         (const uint8_t*)&info));
    file.add_chunk(external::riff::Chunk("PDAT",
                                         2*sizeof(T)*ping.size(),
                                         (const uint8_t*)ping.ping_data()));
    file.add_chunk(external::riff::Chunk("BDAT",
                                         sizeof(float)*ping.bearing_count(),
                                         (const uint8_t*)ping.bearings().data()));
    return true;
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_SONAR_PING_H_
