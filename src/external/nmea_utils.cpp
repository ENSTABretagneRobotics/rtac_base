#include <rtac_base/external/nmea_utils.h>

namespace rtac { namespace nmea {

NmeaError::Code nmea_invalid(const std::string& msg)
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

std::array<double,3> latlonalt_from_gpgga(const std::string& msg, bool checkFormat)
{
    if(checkFormat) { if(auto err = nmea_invalid(msg)) {
        throw NmeaError(err) << " : '" << msg << "'";
    }}

    std::string token;
    std::istringstream iss(msg);

    std::getline(iss, token, ',');
    if(checkFormat && token != "$GPGGA") {
        throw NmeaError() << " : not a GPGGA message '" << msg << "'";
    }
    std::getline(iss, token, ','); // time field is ignored

    std::getline(iss, token, ',');
    double lat = std::stod(token.substr(0, 2)) + std::stod(token.substr(2)) / 60.0;
    std::getline(iss, token, ',');
    if(token[0] == 'S') {
        lat = -lat;
    }

    std::getline(iss, token, ',');
    double lon = std::stod(token.substr(0, 3)) + std::stod(token.substr(3)) / 60.0;
    std::getline(iss, token, ',');
    if(token[0] == 'W') {
        lon = -lon;
    }

    std::getline(iss, token, ','); // quality
    std::getline(iss, token, ','); // stats
    std::getline(iss, token, ','); // hdop

    std::getline(iss, token, ',');
    double alt = std::stod(token);

    return {lat,lon,alt};
}

Pose<double> pose_from_gpgga(const std::string& msg)
{
    auto t = latlonalt_from_gpgga(msg);

    Pose<double> res;
    res.x() = t[0];
    res.y() = t[1];
    res.z() = t[2];
    return res;
}

} //namespace nmea
} //namespace rtac
