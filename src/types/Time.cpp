#include <rtac_base/types/Time.h>

namespace rtac { namespace types {

Time::Time(uint64_t ns) : ns_(ns) {}
Time::Time() : ns_(0) {}
    
Time& Time::operator+=(const Time& other) { ns_ += other.ns_; return *this; }
Time& Time::operator-=(const Time& other) { ns_ -= other.ns_; return *this; }

bool Time::operator==(const Time& other) { return ns_ == other.ns_; }
bool Time::operator< (const Time& other) { return ns_ <  other.ns_; }
bool Time::operator<=(const Time& other) { return ns_ <= other.ns_; }
bool Time::operator> (const Time& other) { return !(*this <= other); }
bool Time::operator>=(const Time& other) { return !(*this <  other); }

}; //namespace types
}; //namespace rtac

rtac::types::Time operator+(const rtac::types::Time& lhs, const rtac::types::Time& rhs)
{
    rtac::types::Time t = lhs;
    t += rhs;
    return t;
}

rtac::types::Time operator-(const rtac::types::Time& lhs, const rtac::types::Time& rhs)
{
    rtac::types::Time t = lhs;
    t -= rhs;
    return t;
}

std::ostream& operator<<(std::ostream& os, const rtac::types::Time& t)
{
    os << t.nanoseconds();
    return os;
}
