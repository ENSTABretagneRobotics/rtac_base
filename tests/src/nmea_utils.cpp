#include <iostream>
#include <list>
using namespace std;

#include <rtac_base/external/nmea_utils.h>
using namespace rtac::nmea;

int main()
{
    // All messages are valid except for the last ones which were edited to
    // exibit various errors.
    std::list<std::string> msgs;
    msgs.push_back("$GPGGA,142042.70,4811.68519184,N,00300.99147341,W,4,15,0.0,121.352,M,49.938,M,1.6,0153*5B\r\n");
    msgs.push_back("$GPVTG,343.49,T,343.22,M,2.810,N,5.204,K,R*35\r\n");
    msgs.push_back("$GPGST,142042.70,,0.024,0.023,0.00,0.024,0.023,0.022*4F\r\n");
    msgs.push_back("$PASHR,142042.70,337.97,T,-001.67,-001.23,-000.00,0.014,0.014,0.065,2,0,*2F\r\n");
    msgs.push_back("$GPGGA,142042.75,4811.68522913,N,00300.99148992,W,4,15,0.0,121.352,M,49.938,M,1.6,0153*5B\r\n");
    msgs.push_back("$GPVTG,343.35,T,343.07,M,2.800,N,5.185,K,R*32\r\n");
    msgs.push_back("$GPGST,142042.75,,0.024,0.023,0.00,0.024,0.023,0.022*4A\r\n");
    msgs.push_back("$PASHR,142042.75,337.89,T,-001.66,-001.24,-000.00,0.014,0.014,0.065,2,0,*23\r\n");
    // First character was deleted. Should trigger an InvalidDelimiter error.
    msgs.push_back("GPGGA,142042.80,4811.68526652,N,00300.99150646,W,4,15,0.0,121.352,M,49.938,M,1.6,0153*50\r\n");
    // Last character was deleted. Should trigger an InvalidDelimiter error.
    msgs.push_back("$GPVTG,343.51,T,343.24,M,2.805,N,5.195,K,R*35\r");
    // Checksum delimiter was deleted. Should trigger a NoChecksum error.
    msgs.push_back("$GPGST,142042.80,,0.024,0.023,0.00,0.024,0.023,0.02240\r\n");
    // First character of the first field was deleted (first field should be
    // 142042.80).  This should trigger a ChecksumMismatch error.
    msgs.push_back("$PASHR,42042.80,337.82,T,-001.65,-001.24,-000.00,0.014,0.014,0.065,2,0,*21\r\n");

    for(auto m : msgs) {
        //cout << m;
        if(auto err = nmea_invalid(m)) {
            cout << "NmeaError " << err << " : " << m << endl;
            continue;
        }
        try {
            auto values = latlonalt_from_gpgga(m);
            std::cout << m;
            std::cout << "    "
                      << values[0] << ' '
                      << values[1] << ' '
                      << values[2] << std::endl;
            auto pose = pose_from_gpgga(m);
            std::cout << pose << std::endl;
        }
        catch(const std::exception& e) {
            std::cout << e.what() << std::endl;
        }
    }

    rtac::Pose<float> p = rtac::Pose<double>();
    
    return 0;
}
