#include <rtac_base/Exception.h>
using namespace rtac;

int main()
{
    //throw Exception();
    //throw Exception("Hello");
    //throw Exception("Hello") << " there !";

    //throw FileError();
    //throw FileError() << " : not found.";
    throw FileError("filename.f") << " : not found.";

    return 0;
}

