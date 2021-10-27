#include <iostream>
using namespace std;

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include "functors_test.h"

template <typename T>
void print_type(std::ostream& os = std::cout) { os << "'print_not_defined'"; }
template<> void print_type<float> (std::ostream& os) { os << "'float'";  }
template<> void print_type<float4>(std::ostream& os) { os << "'float4'"; }

template <class FunctorT>
void print_functor_type(std::ostream& os = std::cout)
{
    os << "(InputT : ";   print_type<typename FunctorT::InputT> (os);
    os << ", OutputT : "; print_type<typename FunctorT::OutputT>(os);
    os << ")" << endl;
}

int main()
{
    int N = 10;
    
    print_functor_type<Vectorize4>();
    print_functor_type<Norm4>();
    print_functor_type<MultiType>();

    //MultiType fm(Norm4(), Vectorize4()); // why this not working ?
    //auto fm  = MultiType(Norm4(), Vectorize4());
    MultiType fm(std::make_tuple(Norm4(), Vectorize4()));
    print_functor_type<decltype(fm)>();
    //cout << std::get<0>(fm.functors_) << endl;
    cout << fm(1.0f) << endl;

    HostVector<float> input(N);
    for(int n = 0; n < N; n++) {
        input[n] = n;
    }

    //auto output = scaling(input, functor::Scaling<float>({2.0f}));

    Saxpy f = Saxpy(functors::Offset<float>({3.0f}), functors::Scaling<float>({2.0f}));
    cout << f(1.0f) << endl;

    auto output = saxpy(input, Saxpy(functors::Offset<float>({3.0f}),
                                     functors::Scaling<float>({2.0f})));

    cout << input  << endl;
    cout << output << endl;

    return 0;
}
