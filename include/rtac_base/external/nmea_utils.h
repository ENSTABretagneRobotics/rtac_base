#ifndef _DEF_RTAC_BASE_NMEA_CODEC_H_
#define _DEF_RTAC_BASE_NMEA_CODEC_H_

#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <rtac_base/Exception.h>
#include <rtac_base/types/Pose.h>

namespace rtac { namespace nmea {

struct NmeaError : public Exception {

    enum Code : uint8_t {
        NoError          = 0x00,
        InvalidDelimiter = 0x01,
        InvalidCharacter = 0x02,
        NoChecksum       = 0x04,
        ChecksumMismatch = 0x08,
        Other            = 0x10
    };

    static std::string code_to_string(const Code& code) {
        switch(code) {
            default:               return "Unknown NmeaError"; break;
            case NoError         : return "NoError";           break;
            case InvalidDelimiter: return "InvalidDelimiter";  break;
            case InvalidCharacter: return "InvalidCharacter";  break;
            case NoChecksum      : return "NoChecksum";        break;
            case ChecksumMismatch: return "ChecksumMismatch";  break;
            case Other           : return "Other";             break;
        }
    }

    NmeaError(const Code& code = Other) :
        Exception(code_to_string(code))
    {}

    NmeaError(const std::string& what_arg) :
        Exception(what_arg)
    {}
};

NmeaError::Code nmea_invalid(const std::string& msg);
inline std::string nmea_type(const std::string& msg) 
{
    return msg.substr(1,5);
}

template <typename T = uint64_t> 
inline T millis_from_gps_time(const std::string& date)
{
    return 1000*(3600*std::stoul(date.substr(0,2))  // hours
                 + 60*std::stoul(date.substr(2,2))  // minutes
                 +    std::stoul(date.substr(4,2))) // seconds
                 +    std::stoul(date.substr(7));   // millis
}

std::array<double,3> latlonalt_from_gpgga(const std::string& msg, bool checkFormat = true);
Pose<double> pose_from_gpgga(const std::string& msg, bool checkFormat = true);

}; //namespace nmea
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os, const rtac::nmea::NmeaError::Code& code)
{
    os << rtac::nmea::NmeaError::code_to_string(code);
    return os;
}

#endif //_DEF_RTAC_BASE_NMEA_CODEC_H_

