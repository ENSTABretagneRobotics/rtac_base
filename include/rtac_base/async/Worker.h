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
    std::future<R> push_front(std::shared_ptr<AsyncFunction<R>>&& f);
    template <class R>
    std::future<R> push_back(std::shared_ptr<AsyncFunction<R>>&& f);
};

template <class R> inline
std::future<R> Worker::push_front(std::shared_ptr<AsyncFunction<R>>&& f)
{
    std::future<R> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(f));
    return res;
}

template <class R> inline
std::future<R> Worker::push_back(std::shared_ptr<AsyncFunction<R>>&& f)
{
    std::future<R> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(f));
    return res;
}

} //namespace async
} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_WORKER_H_
