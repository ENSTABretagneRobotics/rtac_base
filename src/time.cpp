#include <rtac_base/time.h>

namespace rtac { namespace time {

Clock::Clock()
{
    this->reset();
}

void Clock::reset()
{
    t0_ = std::chrono::high_resolution_clock::now();
}

FrameCounter::FrameCounter(int resetCount) :
    resetCount_(resetCount),
    count_(0),
    t0_(std::chrono::high_resolution_clock::now())
{}

float FrameCounter::get() const
{
    auto t = std::chrono::high_resolution_clock::now();
    float res = count_ / std::chrono::duration<double>(t - t0_).count();

    if(count_ >= resetCount_) {
        t0_ = t;
        count_ = 0;
    }
    count_++;

    return res;
}

std::ostream& FrameCounter::print(std::ostream& os) const
{
    float res = this->get();
    if(count_ == 1) {
        os << "Frame rate : " << res << "\r" << std::flush;
    }
    return os;
}

}; //namespace time
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, const rtac::time::Clock& clock)
{
    os << "Clock : " << clock.now() << "s";
    return os;
}

std::ostream& operator<<(std::ostream& os, const rtac::time::FrameCounter& counter)
{
    return counter.print(os);
}
