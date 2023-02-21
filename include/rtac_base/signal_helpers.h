#ifndef _DEF_RTAC_BASE_SINAL_HELPERS_H_
#define _DEF_RTAC_BASE_SINAL_HELPERS_H_

#include <vector>
#include <cmath>

#include <rtac_base/types/Complex.h>

namespace rtac { namespace signal {

/**
 * This generates a set of samples of the sinc function.
 *
 * The function is sampled at a period of oversampling*pi. The returned
 * coefficients make up for roughly 99% of the total energy (integration) of
 * the full sinc function.
 *
 * The most prominent use of this function in the rtac framework involve linear
 * interpolation (texture fetch). The oversampling is therefore important to
 * keep some level of precision.
 *
 * sampling period : pi * oversampling
 * domain          : +- pi * (N - 1) / oversampling, roughly 16*pi
 */
template <typename T>
struct SincFunction
{
    static constexpr T HalfEnergyX = 1.39156;

    protected:

    std::vector<T> x_;
    std::vector<T> y_;
    unsigned int oversampling_;
    T samplingPeriod_;

    public:

    SincFunction(unsigned int oversampling = 8) : 
        x_(16*oversampling),
        y_(x_.size()),
        oversampling_(oversampling),
        samplingPeriod_(M_PI / oversampling)
    {
        auto N = y_.size();
        for(int n = 0; n < N; n++) {
            x_[n] = M_PI * (n - 0.5f*(N - 1)) / oversampling;
            y_[n] = std::sin(x_[n]) / x_[n]; // no need to check for x == 0, never happens
        }
    }

    SincFunction(T span, unsigned int oversampling = 8) : 
        x_(2*((unsigned int)(0.5f*oversampling*span / M_PI) + 1)),
        y_(x_.size()),
        oversampling_(oversampling),
        samplingPeriod_(span / (x_.size() - 1))
    {
        auto N = y_.size();
        for(int n = 0; n < N; n++) {
            x_[n] = span * (((float)n) / (N - 1) - 0.5f);
            y_[n] = std::sin(x_[n]) / x_[n]; // no need to check for x == 0, never happens
        }
    }

    SincFunction(T start, T end, unsigned int oversampling = 8) : 
        SincFunction(end - start, oversampling)
    {
        auto N = y_.size();
        auto span = end - start;
        for(int n = 0; n < N; n++) {
            x_[n] = span * ((float)n) / (N - 1) + start;
            if(abs(x_[n]) < 1.0e-6) {
                y_[n] = 1.0f;
            }
            else {
                y_[n] = std::sin(x_[n]) / x_[n];
            }
        }
    }

    std::size_t size() const { return y_.size();   }
    float sampling_period() const { return samplingPeriod_; }

    const std::vector<T>& domain()   const { return x_; }
    const std::vector<T>& function() const { return y_; }
    
    /**
     * This returns the span of the function domain, given a resolution
     * parameter which corresponds to the domain value x where sin(x) / x =
     * sqrt(0.5)
     */
    T physical_span(T resolution) const {
        T scaling = 2.0 * HalfEnergyX / resolution;
        return 2.0f*x_.back() / scaling;
    }
};

template <typename T>
class SineFunction
{
    protected:

    std::vector<T> x_;
    std::vector<T> y_;
    T              periodCount_;

    public:

    SineFunction(T periodCount, unsigned int oversampling = 8) :
        x_(2*(((unsigned int)periodCount * oversampling) + 1)),
        y_(x_.size()),
        periodCount_(periodCount)

    {
        auto N = x_.size();
        for(int n = 0; n < N; n++) {
            x_[n] = (2.0*M_PI*periodCount_*n) / (N - 1);
            y_[n] = std::sin(x_[n]);
        }
    }

    std::size_t size() const { return y_.size(); }

    const std::vector<T>& phase()    const { return x_; }
    const std::vector<T>& function() const { return y_; }
};

template <typename T>
class ComplexSineFunction
{
    protected:

    std::vector<T> x_;
    std::vector<Complex<T>> y_;
    T              periodCount_;

    public:

    ComplexSineFunction(T periodCount, unsigned int oversampling = 8) :
        x_(2*(((unsigned int)periodCount * oversampling) + 1)),
        y_(x_.size()),
        periodCount_(periodCount)

    {
        auto N = x_.size();
        for(int n = 0; n < N; n++) {
            x_[n] = (2.0*M_PI*periodCount_*n) / (N - 1);
            y_[n] = Complex<T>(std::cos(x_[n]), std::sin(x_[n]));
        }
    }

    std::size_t size() const { return y_.size(); }

    const std::vector<T>&          phase()    const { return x_; }
    const std::vector<Complex<T>>& function() const { return y_; }
};



} //namespace signal
} //namespace rtac

#endif //_DEF_RTAC_BASE_SINAL_HELPERS_H_
