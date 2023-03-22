#ifndef _DEF_RTAC_BASE_ASYNC_ASYNC_WORKER_H_
#define _DEF_RTAC_BASE_ASYNC_ASYNC_WORKER_H_

#include <thread>
#include <future>
#include <functional>
#include <memory>
#include <deque>

#include <rtac_base/async/AsyncFunction.h>

namespace rtac {

class AsyncWorker
{
    protected:

    std::deque<AsyncFunctionBase::Ptr> nextQueue_;
    std::deque<AsyncFunctionBase::Ptr> workQueue_;
    std::mutex                         queueLock_; // for operations on queues

    bool execute_next_queue();

    public:

    AsyncWorker();

    void run();

    std::future<void> push_front(EmptyAsyncFunction::Ptr&& f);
    std::future<void> push_back( EmptyAsyncFunction::Ptr&& f);

    template <class R>
    std::future<R> push_front(std::unique_ptr<AsyncFunction<R>>&& f);
    template <class R>
    std::future<R> push_back(std::unique_ptr<AsyncFunction<R>>&& f);

    template <class R, class... Args1, class... Args2>
    R execute(R(*f)(Args1...), Args2&&... args);
};

template <class R> inline
std::future<R> AsyncWorker::push_front(std::unique_ptr<AsyncFunction<R>>&& f)
{
    std::future<R> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_front(std::move(f));
    return res;
}

template <class R> inline
std::future<R> AsyncWorker::push_back(std::unique_ptr<AsyncFunction<R>>&& f)
{
    std::future<R> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(f));
    return res;
}

template <class R, class... Args1, class... Args2> inline
R AsyncWorker::execute(R(*f)(Args1...), Args2&&... args)
{
    auto res = this->push_back(async_bind(f, args...));
    res.wait();
    return res.get();
}

} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_WORKER_H_
