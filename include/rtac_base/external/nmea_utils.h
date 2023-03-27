#ifndef _DEF_RTAC_BASE_NMEA_CODEC_H_
#define _DEF_RTAC_BASE_NMEA_CODEC_H_

#include <string>
#include <iostream>
#include <iomanip>

#include <rtac_base/types/common.h>

namespace rtac { namespace nmea {

struct NmeaError : public std::runtime_error {

    enum Code : uint8_t {
        NoError          = 0x0,
        InvalidDelimiter = 0x1,
        InvalidCharacter = 0x2,
        NoChecksum       = 0x4,
        ChecksumMismatch = 0x8,
    };

    static std::string code_to_string(const Code& code) {
        switch(code) {
            default:               return "Unknown NmeaError"; break;
            case NoError         : return "NoError";           break;
            case InvalidDelimiter: return "InvalidDelimiter";  break;
            case InvalidCharacter: return "InvalidCharacter";  break;
            case NoChecksum      : return "NoChecksum";        break;
            case ChecksumMismatch: return "ChecksumMismatch";  break;
        }
    }

    NmeaError(const Code& code) :
        std::runtime_error(code_to_string(code))
    {}

    NmeaError(const std::string& what_arg) :
        std::runtime_error(what_arg)
    {}
};

inline NmeaError::Code nmea_invalid(const std::string& msg)
{
    if(msg[0] != '$' || msg[msg.size() - 2] != '\r' || msg[msg.size() - 1] != '\n') {
        return NmeaError::InvalidDelimiter;
    }

    uint8_t checksum = 0;
    unsigned int i = 1;
    for(; i < msg.size(); i++) {
        if(msg[i] == '*') {
            i++;
            break;
        }
        if(msg[i] < 32 || msg[i] > 126) {
            if(msg[i] == '\r') {
                // End of message reached without finding a checksum delimiter.
                return NmeaError::NoChecksum;
            }
            return NmeaError::InvalidCharacter;
        }
        checksum ^= msg[i];
    }
    
    auto transmittedChecksum = std::stoul(msg.c_str() + i, nullptr, 16);
    if(transmittedChecksum != checksum) {
        return NmeaError::ChecksumMismatch;
    }
    return NmeaError::NoError;
}

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

}; //namespace nmea
}; //namespace rtac

inline std::ostream& operator<<(std::ostream& os, const rtac::nmea::NmeaError::Code& code)
{
    os << rtac::nmea::NmeaError::code_to_string(code);
    return os;
}

#endif //_DEF_RTAC_BASE_NMEA_CODEC_H_

