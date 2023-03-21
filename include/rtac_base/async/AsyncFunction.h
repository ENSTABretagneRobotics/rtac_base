#ifndef _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
#define _DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_

#include <functional>
#include <future>

namespace rtac { namespace async {

/**
 * This is a base type for the async::AsyncFunction<T> type.
 * 
 * Its may purpose is to provide an abstract interface to async::AsyncFunction
 * to be able to store them in a single container.
 */
struct AsyncFunctionBase
{
    using Ptr = std::shared_ptr<AsyncFunctionBase>;

    void operator()() { this->execute(); }

    virtual void execute() = 0;
};

/**
 * This stores a std::function to be executed at a later time and a
 * std::promise to get the result.
 *
 * The std::promise object acts as a synchonization primitive to wait for the
 * result.
 */
template <class R>
class AsyncFunction : public AsyncFunctionBase
{
    public:

    using Ptr = std::shared_ptr<AsyncFunction<R>>;
    using result_type = R;

    protected:

    std::function<R(void)> function_;
    std::promise<R>        promise_;

    public:

    AsyncFunction() = delete;
    AsyncFunction(const AsyncFunction<R>&) = delete;
    AsyncFunction<R>& operator=(const AsyncFunction<R>&) = delete;

    AsyncFunction(std::function<R(void)>&& f) : 
        function_(std::move(f))
    {}

    AsyncFunction(AsyncFunction<R>&& f) :
        function_(std::move(f.function_)),
        promise_ (std::move(f.promise_))
    {}
    AsyncFunction<R>& operator=(AsyncFunction<R>&& f) {
        function_ = std::move(f.function_);
        promise_  = std::move(f.promise_);
    }

    std::future<R> future() { return promise_.get_future(); }
    void execute() { promise_.set_value(function_());   }
};

/**
 * This is a specialization of AsyncFunction for void return type. It is
 * identical except for the execute method which has to be modified because we
 * cannot call the std::promise::set_value method directly on the result of a
 * void function.
 *
 * The std::promise object acts as a synchonization primitive to wait for the
 * result.
 */
template <>
class AsyncFunction<void> : public AsyncFunctionBase
{
    public:

    using Ptr = std::shared_ptr<AsyncFunction<void>>;
    using result_type = void;

    protected:

    std::function<void(void)> function_;
    std::promise<void>        promise_;

    public:

    AsyncFunction() = delete;
    AsyncFunction(const AsyncFunction<void>&) = delete;
    AsyncFunction<void>& operator=(const AsyncFunction<void>&) = delete;

    AsyncFunction(std::function<void(void)>&& f) : 
        function_(std::move(f))
    {}

    AsyncFunction(AsyncFunction<void>&& f) :
        function_(std::move(f.function_)),
        promise_ (std::move(f.promise_))
    {}
    AsyncFunction<void>& operator=(AsyncFunction<void>&& f) {
        function_ = std::move(f.function_);
        promise_  = std::move(f.promise_);
    }

    std::future<void> future() { return promise_.get_future(); }
    void execute() { function_(); promise_.set_value();        }
};

/**
 * This is a special specialization of AsyncFunctionBase. The only goal is to
 * provide a synchronization primitive and does nothing otherwise
 */
class EmptyAsyncFunction : public AsyncFunctionBase
{
    public:

    using Ptr = std::shared_ptr<EmptyAsyncFunction>;
    using result_type = void;

    protected:

    std::promise<void> promise_;

    public:

    EmptyAsyncFunction() {}

    std::future<void> future() { return promise_.get_future(); }
    void execute()             { promise_.set_value();         }
};

template <class R> inline typename 
AsyncFunction<R>::Ptr make_async(std::function<R(void)>&& f)
{
    return std::make_shared<AsyncFunction<R>>(std::forward<std::function<R(void)>>(f));
}

inline EmptyAsyncFunction::Ptr make_empty_async()
{
    return std::make_shared<EmptyAsyncFunction>();
}

} //namespace async
} //namespace rtac

#endif //_DEF_RTAC_BASE_ASYNC_ASYNC_FUNCTION_H_
