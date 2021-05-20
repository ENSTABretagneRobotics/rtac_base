#include <rtac_base/time.h>

namespace rtac { namespace time {

Clock::Clock()
{
    this->reset();
}

/**
 * Set reference epoch t0_ to now.
 */
void Clock::reset()
{
    t0_ = std::chrono::high_resolution_clock::now();
}

FrameCounter::FrameCounter(int resetCount) :
    resetCount_(resetCount),
    count_(0),
    t0_(std::chrono::high_resolution_clock::now()),
    period_(Duration::zero())
{}

float FrameCounter::get() const
{
    auto t = std::chrono::high_resolution_clock::now();
    Duration ellapsed(t - t0_);
    float res = count_ / ellapsed.count();

    if(period_ != Duration::zero()) {
        while(ellapsed < period_) {
            std::this_thread::sleep_for(period_ - ellapsed);
            ellapsed = (std::chrono::high_resolution_clock::now() - t0_);
        }
    }

    if(count_ >= resetCount_) {
        t0_ = t;
        count_ = 0;
    }
    count_++;

    return res;
}

void FrameCounter::limit_frame_rate(float fps)
{
    if(fps > 1.0e-8)
        period_ = Duration(1.0 / fps);
}

void FrameCounter::free_frame_rate()
{
    period_ = Duration::zero();
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
