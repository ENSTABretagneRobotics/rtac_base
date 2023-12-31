#ifndef _DEF_RTAC_BASE_EXCEPTION_H_
#define _DEF_RTAC_BASE_EXCEPTION_H_

#include <exception>
#include <string>
#include <sstream>

namespace rtac {

class Exception : public std::exception
{
    protected:

    std::string err_;

    public:

    Exception(const std::string& err = "RTAC_ERROR") :
        err_(err)
    {}
    Exception& operator=(const Exception&) = default;

    virtual const char* what() const noexcept { return err_.c_str(); }

    template <typename T>
    Exception& operator<<(const T& rhs) {
        std::ostringstream oss;
        oss << err_ << rhs;
        err_ = oss.str();
        return *this;
    }
};

struct FileError : public Exception
{
    FileError() : Exception("RTAC_FILE_ERROR") {}
    FileError(const std::string& filename) : 
        Exception("RTAC_FILE_ERROR '" + filename + "'")
    {}
};

struct FormatError : public Exception
{
    FormatError() : Exception("RTAC_FORMAT_ERROR") {}
};

} //namespace rtac


#endif //_DEF_RTAC_BASE_EXCEPTION_H_
