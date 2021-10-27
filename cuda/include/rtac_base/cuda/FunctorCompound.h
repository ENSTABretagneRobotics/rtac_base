#ifndef _DEF_RTAC_BASE_CUDA_FUNCTOR_COMPOUND_H_
#define _DEF_RTAC_BASE_CUDA_FUNCTOR_COMPOUND_H_

#include <tuple>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace cuda { namespace functors {

/**
 * This class allows for the creation of custom unary Functor types on the fly.
 *
 * A functor is a callable struct (defines an operator()). In the RTAC
 * framework, a valid functor must define an InputT and OutputT types, as well
 * as the operator(). As such, a minimal functor code has the following form :
 *
 * \code
 * struct MultiplyBy2 {
 *     using InputT  = float;
 *     using OutputT = float;
 *     
 *     float operator()(float input) const { return 2.0f * input; }
 * };
 * \endcode
 *
 * Functors can be templates :
 *
 * \code
 * template <typename T>
 * struct MultiplyBy2 {
 *     using InputT  = T;
 *     using OutputT = T;
 *     
 *     T operator()(T input) const { return 2.0f * input; }
 * };
 * \endcode
 *
 * Combining two functors can be done like so :
 *
 * \code
 * auto multBy2ThenAdd3 = FunctorCompound(Offset(3), Scaling(2));
 * \endcode
 *
 * After compilation in release mode, the compound is equivalent to directly
 * writting the operation by hand (but with the benefit is has been written at
 * a single location in the code).
 */
template <class... FunctorsT>
struct FunctorCompound
{
    using TupleT = std::tuple<FunctorsT...>;
    static constexpr unsigned int FunctorCount = std::tuple_size<TupleT>::value;
    static constexpr unsigned int LastIndex    = FunctorCount - 1;
    
    template <unsigned int Level>
    struct functor_get {
        using type    = typename std::tuple_element<Level,TupleT>::type;
        using InputT  = typename type::InputT;
        using OutputT = typename type::OutputT;
    };

    using InputT  = typename functor_get<LastIndex>::InputT;
    using OutputT = typename functor_get<0>::OutputT;

    TupleT functors_;

    template <unsigned int Level> RTAC_HOSTDEVICE
    typename functor_get<Level>::OutputT call_functor(const InputT& input) const {
        if constexpr(Level == LastIndex) {
            return std::get<Level>(functors_)(input);
        }
        else {
            return std::get<Level>(functors_)(call_functor<Level+1>(input));
        }
        // CAUTION : THE CODE BELOW IS UNREACHABLE, BUT THIS IS DONE ON
        // PURPOSE.
        // At the time this file were written, there was a bug in nvcc compiler
        // about if constexpr. The bug triggers a "warning: missing return
        // statement at end of non-void function" even though the function
        // always returns in one of the branch of the condition above. The line
        // below is to suppress the warning but has no effect on the code. See
        // here for more info :
        // https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
        return std::get<Level>(functors_)(input);
    }

    public:
    
    constexpr FunctorCompound(const TupleT& functors) : functors_(functors) {}
    constexpr FunctorCompound(FunctorsT... functors) : functors_(std::make_tuple(functors...)) {}

    RTAC_HOSTDEVICE OutputT operator()(const InputT& input) const {
        return call_functor<0>(input);
    }
};

}; //namespace functors
}; //namespace cuda
}; //namespace rtac

#endif //_DEF_RTAC_BASE_CUDA_FUNCTOR_COMPOUND_H_
