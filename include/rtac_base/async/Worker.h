#ifndef _DEF_RTAC_BASE_ASYNC_WORKER_H_
#define _DEF_RTAC_BASE_ASYNC_WORKER_H_

#include <thread>
#include <future>
#include <functional>
#include <memory>
#include <deque>

#include <rtac_base/async/AsyncFunction.h>

namespace rtac { namespace async {

class Worker
{
    protected:

    std::deque<AsyncFunctionBase::Ptr> nextQueue_;
    std::deque<AsyncFunctionBase::Ptr> workQueue_;
    std::mutex                         queueLock_; // for operations on queues

    //std::future<void> stopBarrier_;
    //bool              shouldStop_;

    bool execute_next_queue();

    public:

    Worker();

    void run();

    void add_callback(std::function<void(void)>&& f);
    void add_callback(void(*f)(void));

    template <typename R>
    std::future<R> add_callback(std::function<R(void)>&& f);
    template <typename R>
    std::future<R> add_callback(R(*f)(void));
};

template <typename R> inline
std::future<R> Worker::add_callback(std::function<R(void)>&& f)
{
    auto asyncF = make_async_function(std::forward<std::function<R(void)>>(f));
    std::future<R> res = asyncF->future();
    {
        std::lock_guard<std::mutex> lock(queueLock_);
        nextQueue_.push_back(std::move(asyncF));
    }
    return res;
}

template <typename R> inline
std::future<R> Worker::add_callback(R(*f)(void))
{
    this->add_callback(std::move(std::function<R(void)>(f)));
}

} //namespace async
} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_WORKER_H_
