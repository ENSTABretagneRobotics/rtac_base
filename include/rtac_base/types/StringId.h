#ifndef _DEF_RTAC_BASE_TYPES_STRING_ID_H_
#define _DEF_RTAC_BASE_TYPES_STRING_ID_H_

#include <iosfwd>
#include <string>
#include <cstring>

#include <rtac_base/common.h>
#include <rtac_base/Exception.h>

namespace rtac {

#pragma pack(push,1)
/**
 * The StringId type is to be used for identifiers in serialized data.
 * 
 * The StringId size in memory is strictly equal to the usable number of
 * characters. As such, it may or may not contain a '\0' terminated string.  If
 * the '\0' is present in the array, it is valid as a null terminated string
 * and only the character before are considered (as with regular C strings).
 */
template <unsigned int N>
struct StringId
{
    static constexpr unsigned int Size = N;

    char data[Size];

    StringId()                           = default;
    StringId(const StringId&)            = default;
    StringId& operator=(const StringId&) = default;

    StringId(const std::string& id) 
    {
        if(id.size() > Size) {
            throw Exception("StrindId") << '<' << Size
                << "> : input string too large (got " << id.size()
                << ", must be < " << Size << ')';
        }
        std::memcpy(data, id.c_str(), id.size());
        if(id.size() < Size) {
            data[id.size()] = '\0';
        }
    }

    std::string string()   const { return std::string(data, Size); }
    operator std::string() const { return this->string();          }

    bool operator==(const char* id) const {
        for(unsigned int i = 0; i < Size; i++) {
            if(data[i] == '\0' || id[i] == '\0') {
                return data[i] == id[i];
            }
            if(data[i] != id[i]) {
                return false;
            }
        }
        return id[Size] == '\0';
    }
    bool operator!=(const char* id)        const { return !(*this == id);         }
    bool operator==(const std::string& id) const { return *this == id.c_str();    }
    bool operator!=(const std::string& id) const { return !(*this == id);         }
    bool operator==(const StringId& other) const { return *this == other.data;    }
    bool operator!=(const StringId& other) const { return !(*this == other.data); }
    template <unsigned int N2>
    bool operator==(const StringId<N2>& other) const { return *this == other.string(); }
    template <unsigned int N2>
    bool operator!=(const StringId<N2>& other) const { return !(*this == other);       }
};
#pragma pack(pop)

// mandatory in C++14, relaxed in C++17
template <unsigned int N> constexpr unsigned int StringId<N>::Size;

} //namespace rtac

template <unsigned int Size> inline
std::ostream& operator<<(std::ostream& os, const rtac::StringId<Size>& id)
{
    os << id.string();
    return os;
}

#endif //_DEF_RTAC_BASE_TYPES_STRING_ID_H_
