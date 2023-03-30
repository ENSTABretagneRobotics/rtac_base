#ifndef _DEF_RTAC_BASE_EXTERNAL_PROJ_H_
#define _DEF_RTAC_BASE_EXTERNAL_PROJ_H_

#include <string>

#include <proj.h>

#include <rtac_base/Exception.h>

namespace rtac { namespace external {

class ProjTransform
{
    template <typename T>
    struct ENU { 
        ENU() = default;
        ENU(T e, T n, T u) : east(e), north(n), up(u) {}
        T east; T north; T up;
    };

    protected:

    std::string fromStr_;
    std::string toStr_;

    PJ_CONTEXT* context_;
    PJ*         transformation_;

    public:

    ProjTransform(const std::string& from, const std::string& to);
    ~ProjTransform();

    PJ_COORD forward (const PJ_COORD& from) const;
    PJ_COORD backward(const PJ_COORD& from) const;

    template <typename T> ENU<T> forward (const ENU<T>& enu) const;
    template <typename T> ENU<T> backward(const ENU<T>& enu) const;
    
    template <typename T> ENU<T> forward (T east, T north, T up = 0.0) const;
    template <typename T> ENU<T> backward(T east, T north, T up = 0.0) const;
};

template <typename T>
ProjTransform::ENU<T> ProjTransform::forward(const ENU<T>& enu) const
{
    PJ_COORD p = this->forward(proj_coord((double)enu.east,
                                          (double)enu.north,
                                          (double)enu.up, 0.0));
    return ENU<T>(p.enu.e, p.enu.n, p.enu.u);
}

template <typename T>
ProjTransform::ENU<T> ProjTransform::backward(const ENU<T>& enu) const
{
    PJ_COORD p = this->backward(proj_coord((double)enu.east,
                                           (double)enu.north,
                                           (double)enu.up, 0.0));
    return ENU<T>(p.enu.e, p.enu.n, p.enu.u);
}

template <typename T>
ProjTransform::ENU<T> ProjTransform::forward(T east, T north, T up) const
{
    PJ_COORD p = this->forward(proj_coord((double)east,
                                          (double)north,
                                          (double)up, 0.0));
    return ENU<T>(p.enu.e, p.enu.n, p.enu.u);
}

template <typename T>
ProjTransform::ENU<T> ProjTransform::backward(T east, T north, T up) const
{
    PJ_COORD p = this->backward(proj_coord((double)east,
                                           (double)north,
                                           (double)up, 0.0));
    return ENU<T>(p.enu.e, p.enu.n, p.enu.u);
}

} //namespace external
} //namespace rtac

template <typename T> inline
std::ostream& operator<<(std::ostream& os, const rtac::external::ProjTransform::ENU<T>& enu)
{
    os << "(east: "   << enu.east
       << ", north: " << enu.north
       << ", up: "    << enu.up << ')';
    return os;
}

#endif //_DEF_RTAC_BASE_EXTERNAL_PROJ_H_
