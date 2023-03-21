#include <iostream>
#include <functional>
using namespace std;

#include <rtac_base/async/Worker.h>
using namespace rtac::async;

void hello_there()
{
    std::cout << "Hello there !" << std::endl;
}

void hello(Worker* worker, unsigned int count)
{
    if(count == 0)
        return;
    std::cout << "Hello " << count << '!' << std::endl;
    worker->add_callback(std::bind(hello, worker, count - 1));
}

int main()
{
    Worker worker;
    
    worker.add_callback(hello_there);
    worker.add_callback(std::bind(hello, &worker, 5));
    worker.run();

    return 0;
}
