#ifndef _DEF_RTAC_BASE_MISC_H_
#define _DEF_RTAC_BASE_MISC_H_

#include <iostream>
#include <chrono>
#include <thread>

namespace rtac { namespace time {

/** 
 * Simple type to measure time less verbose than std::chrono.
 */
class Clock
{
    protected:

    std::chrono::time_point<std::chrono::high_resolution_clock> t0_;

    public:

    Clock();
    
    void reset();
    
    /**
     * Return current time relative to epoch t0_.
     * (Successive calls to interval will give ellapsed time since last reset).
     */
    template<typename T = double>
    T now() const
    {
        return std::chrono::duration<T>(
            std::chrono::high_resolution_clock::now() - t0_).count();
    } 

    /**
     * Return current time relative to epoch t0_, and set epoch t0_ to now.
     * (Successive calls to interval will give ellapsed time since last call).
     */
    template<typename T = double>
    T interval()
    {
        T res;
        auto t = std::chrono::high_resolution_clock::now();
        res = std::chrono::duration<T>(t - t0_).count();
        t0_ = t;

        return res;
    } 
};

/** 
 * Simple type to count a frequency (in Hz or FramePerSecond).
 */
class FrameCounter
{
    public:

    using Duration = std::chrono::duration<double, std::ratio<1,1>>;

    protected:
    
    int resetCount_;
    mutable int count_;
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> t0_;
    Duration period_;

    public:
    
    FrameCounter(int resetCount = 1);

    void limit_frame_rate(float fps);
    void free_frame_rate();
    
    float get() const;
    std::ostream& print(std::ostream& os) const;
};

}; //namespace time
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::time::Clock& clock);
std::ostream& operator<<(std::ostream& os, const rtac::time::FrameCounter& counter);

#endif //_DEF_RTAC_BASE_MISC_H_
