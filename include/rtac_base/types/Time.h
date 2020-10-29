#ifndef _DEF_RTAC_BASE_TYPES_TIME_H_
#define _DEF_RTAC_BASE_TYPES_TIME_H_

#include <iostream>
#include <chrono>

namespace rtac { namespace types {

struct Time
{
    protected:

    uint64_t ns_;

    Time(uint64_t ns) : ns_(ns) {}

    public:

    Time() : ns_(0) {}
    
    template <typename T>
    static Time from_nanoseconds(const T& ns) { return Time(static_cast<uint64_t>(ns)); }
    template <typename T>
    static Time from_microseconds(const T& us) { return Time(1000*static_cast<uint64_t>(us)); }
    template <typename T>
    static Time from_milliseconds(const T& ms) { return Time(1000000*static_cast<uint64_t>(ms)); }
    template <typename T>
    static Time from_seconds(const T& s) { return Time(1000000000*static_cast<uint64_t>(s)); }

    template <typename T = uint64_t>
    T nanoseconds() const { return ns_; }
    template <typename T = uint64_t>
    T microseconds() const { return ns_ / static_cast<T>(1000); }
    template <typename T = uint64_t>
    T milliseconds() const { return ns_ / static_cast<T>(1000000); }
    template <typename T = double>
    T seconds() const { return ns_ / static_cast<T>(1000000000); }

    Time& operator+=(const Time& other) { ns_ += other.ns_; return *this; }
    Time& operator-=(const Time& other) { ns_ -= other.ns_; return *this; }

    bool operator==(const Time& other) { return ns_ == other.ns_; }
    bool operator< (const Time& other) { return ns_ <  other.ns_; }
    bool operator<=(const Time& other) { return ns_ <= other.ns_; }
    bool operator> (const Time& other) { return !(*this <= other); }
    bool operator>=(const Time& other) { return !(*this <  other); }
};

template <typename ClockT = std::chrono::steady_clock>
class StandardClock
{
    public:

    using ClockType = ClockT;
    
    static Time now()
    {
        return Time::from_nanoseconds(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                ClockType::now().time_since_epoch()).count());
    }
};

template <typename ClockT = StandardClock<>>
class Clock
{
    public:

    using ClockType = ClockT;
    
    static Time now() { return ClockType::now(); }
};

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

#endif //_DEF_RTAC_BASE_TYPES_TIME_H_


