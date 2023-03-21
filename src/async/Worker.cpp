#include <rtac_base/async/Worker.h>

namespace rtac { namespace async {

Worker::Worker()
{}

void Worker::run()
{
    while(execute_next_queue());
}

bool Worker::execute_next_queue()
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

void Worker::add_callback(std::function<void(void)>&& f)
{
    std::lock_guard<std::mutex> lock(queueLock_);
    nextQueue_.push_back(std::move(make_async_function(std::forward<std::function<void()>>(f))));
}

} //namespace async
} //namespace rtac
