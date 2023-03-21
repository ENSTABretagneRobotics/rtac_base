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

    bool execute_next_queue();

    public:

    Worker();

    void run();

    template <class R>
    std::future<R> push_back(typename AsyncFunction<R>::Ptr&& f);

    template <typename R>
    std::future<R> push_back(std::function<R(void)> f);
    template <typename R>
    std::future<R> push_back(R(*f)(void));
};

template <class R> inline
std::future<R> Worker::push_back(typename AsyncFunction<R>::Ptr&& f)
{
    std::future<R> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(f));
    return res;
}

template <typename R> inline
std::future<R> Worker::push_back(std::function<R(void)> f)
{
    return this->push_back<R>(std::move(make_async(std::forward<std::function<R(void)>>(f))));
}

template <typename R> inline
std::future<R> Worker::push_back(R(*f)(void))
{
    return this->push_back<R>(std::move(std::function<R(void)>(f)));
}

} //namespace async
} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_WORKER_H_
