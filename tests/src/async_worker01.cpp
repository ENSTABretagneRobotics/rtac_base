#include <iostream>
#include <sstream>
#include <thread>
#include <functional>
using namespace std;

#include <rtac_base/async/AsyncWorker.h>
using namespace rtac;

void hello_there()
{
    std::cout << "Hello there !" << std::endl;
    getchar();
}

unsigned int hello(AsyncWorker* worker, unsigned int count)
{
    if(count == 0)
        return 0;
    
    if(count == 3)
        getchar();

    std::cout << "Hello " << count << '!' << std::endl;
    worker->push_back(async_bind(hello, worker, count - 1));
    return count;
}

template <typename T>
void future_type(const std::future<T>&) {
    std::cout << "std::future<T>" << std::endl;
}

void future_type(const std::future<void>&) {
    std::cout << "std::future<void>" << std::endl;
}

struct Test
{
    float get() {
        std::cout << "Test::get() called" << std::endl;
        return 14.0;
    }
    float get() const {
        std::cout << "Test::get() const called" << std::endl;
        return 42.0;
    }
    float get_const() const {
        std::cout << "Test::get_const() const called" << std::endl;
        getchar();
        return 314.0;
    }
};

unsigned int echo(unsigned int v) { return v; }

int main()
{
    AsyncWorker worker;
    
    worker.push_back(async_bind(hello_there));
    auto res0 = worker.push_back(async_bind(hello, &worker, 5));

    Test t0;
    auto res1 = worker.push_back(async_bind(&Test::get, &t0));
    auto res2 = worker.push_back(async_bind(&Test::get, (const Test*)&t0));
    auto res4 = worker.push_front(empty_async());
    auto res3 = worker.push_front(async_bind(&Test::get_const, &t0));
    auto res5 = worker.push_back(async_bind([](int v){ std::cout << "lambda called : " << v << std::endl; return v; }, 89));

    std::thread th(std::bind(&AsyncWorker::run, &worker));

    std::ostringstream oss0;
    oss0 << "sync lambda executed : " << worker.execute([](int v){ return v + 10; }, 101);
    std::cout << oss0.str() << std::endl;

    res4.wait();
    std::cout << "There !" << std::endl;

    std::ostringstream oss;
    oss << "execution result : " << worker.execute(echo, 74);
    std::cout << oss.str() << std::endl;
    
    future_type(res0);
    res0.wait();
    std::cout << "Result : " << res0.get() << std::endl;

    res1.wait();
    std::cout << "Test::get result : " << res1.get() << std::endl;

    res2.wait();
    std::cout << "Test::get const result : " << res2.get() << std::endl;

    res3.wait();
    std::cout << "Test::get const result : " << res3.get() << std::endl;
    
    th.join();

    return 0;
}
