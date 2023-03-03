#include <rtac_base/Exception.h>

int main()
{
    //throw rtac::Exception();
    //throw rtac::Exception("Hello");
    throw rtac::Exception("Hello") << " there !";

    return 0;
}

