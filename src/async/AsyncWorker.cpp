#include <rtac_base/async/AsyncWorker.h>

namespace rtac {

AsyncWorker::AsyncWorker()
{}

void AsyncWorker::run()
{
    while(execute_next_queue());
}

bool AsyncWorker::execute_next_queue()
{
    {
        std::lock_guard<std::mutex> lock(queueLock_);
        if(nextQueue_.size() == 0) {
            return false;
        }
        workQueue_.swap(nextQueue_);
    }

    for(auto& f : workQueue_) {
        f->execute();
    }

    workQueue_.clear();
    return true;
}

std::future<void> AsyncWorker::push_front(EmptyAsyncFunction::Ptr&& f)
{
    std::future<void> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_front(std::move(f));
    return res;
}

std::future<void> AsyncWorker::push_back(EmptyAsyncFunction::Ptr&& f)
{
    std::future<void> res = f->future();
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(f));
    return res;
}

} //namespace rtac
