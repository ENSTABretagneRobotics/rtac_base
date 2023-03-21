#include <iostream>
#include <thread>
#include <functional>
using namespace std;

#include <rtac_base/async/Worker.h>
using namespace rtac::async;

namespace rtac {

template <class R, class... Args>
std::function<R()> bind(R(*f)(Args...), Args... args)
{
    return std::bind(f, args...);
}

template <class R, class C, class... Args>
std::function<R()> bind(R(C::*f)(Args...), C* c, Args... args)
{
    return std::bind(f, c, args...);
}

}


void hello_there()
{
    std::cout << "Hello there !" << std::endl;
}

unsigned int hello(Worker* worker, unsigned int count)
{
    if(count == 0)
        return 0;
    std::cout << "Hello " << count << '!' << std::endl;
    //worker->push_back(std::bind(hello, worker, count - 1));
    worker->push_back(rtac::bind(hello, worker, count - 1));
    return count;
}

template <typename T>
void future_type(const std::future<T>&) {
    std::cout << "std::future<T>" << std::endl;
}

void future_type(const std::future<void>&) {
    std::cout << "std::future<void>" << std::endl;
}

int main()
{
    Worker worker;
    
    worker.push_back(hello_there);

    //auto res = worker.push_back(std::bind(hello, &worker, 5));
    auto res = worker.push_back(rtac::bind(hello, &worker, 5u));

    std::thread th(rtac::bind(&Worker::run, &worker));
    
    future_type(res);
    res.wait();
    //unsigned int value = res.get();
    //std::cout << "Result : " << res.get() << std::endl;
    
    th.join();

    return 0;
}
